use common::utok;
use log::LevelFilter;
use simple_logger::SimpleLogger;
use std::{
    alloc::Layout,
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Mutex,
    time::Instant,
};
use tokenizer::{Tokenizer, BPE};
use transformer_cpu::{
    model_parameters::{Allocator, Llama2, Memory},
    Transformer,
};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Prompt.
    #[clap(short, long)]
    prompt: String,
    /// Max steps.
    #[clap(short, long)]
    step: Option<usize>,
    /// Copy model parameters inside memory.
    #[clap(long)]
    inside_mem: bool,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(short, long)]
    log: Option<String>,

    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

struct NormalAllocator(Mutex<HashMap<*const u8, usize>>);

impl Allocator for NormalAllocator {
    unsafe fn allocate(&self, size: usize) -> NonNull<u8> {
        let ptr = NonNull::new(std::alloc::alloc(Layout::from_size_align_unchecked(
            size,
            std::mem::align_of::<usize>(),
        )))
        .unwrap();
        self.0.lock().unwrap().insert(ptr.as_ptr(), size);
        ptr
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>) {
        std::alloc::dealloc(
            ptr.as_ptr(),
            Layout::from_size_align_unchecked(
                self.0
                    .lock()
                    .unwrap()
                    .remove(&ptr.as_ptr().cast_const())
                    .unwrap(),
                std::mem::align_of::<usize>(),
            ),
        )
    }
}

impl GenerateArgs {
    pub fn invoke(self) {
        let log = self
            .log
            .and_then(|log| match log.to_lowercase().as_str() {
                "off" | "none" => Some(LevelFilter::Off),
                "trace" => Some(LevelFilter::Trace),
                "debug" => Some(LevelFilter::Debug),
                "info" => Some(LevelFilter::Info),
                "error" => Some(LevelFilter::Error),
                _ => None,
            })
            .unwrap_or(LevelFilter::Warn);
        SimpleLogger::new().with_level(log).init().unwrap();

        let model_dir = PathBuf::from(self.model);
        let step = self.step.unwrap_or(usize::MAX);

        let time = Instant::now();
        let tokenizer = BPE::from_model_file(model_dir.join("tokenizer.model")).unwrap();
        info!("build tokenizer ... {:?}", time.elapsed());

        if self.nvidia {
            on_nvidia_gpu(model_dir, tokenizer, self.prompt, step)
        } else {
            on_host(model_dir, tokenizer, self.prompt, step, self.inside_mem)
        }
    }
}

fn on_host(
    model_dir: impl AsRef<Path>,
    tokenizer: impl Tokenizer,
    prompt: impl AsRef<str>,
    step: usize,
    inside_mem: bool,
) {
    let model_dir = model_dir.as_ref();
    let prompt = prompt.as_ref();

    let time = Instant::now();
    let mut model = Box::new(Memory::load_safetensors_from_dir(&model_dir).unwrap());
    info!("load model ... {:?}", time.elapsed());

    if inside_mem {
        let time = Instant::now();
        let allocator = NormalAllocator(Mutex::new(HashMap::new()));
        model = Box::new(Memory::realloc_with(&*model, allocator));
        info!("copy model ... {:?}", time.elapsed());
    }
    let step = step.min(model.max_position_embeddings());
    let time = Instant::now();
    let mut transformer = Transformer::new(model);
    let mut kv_cache = transformer.new_cache();
    info!("build transformer ... {:?}", time.elapsed());

    let time = Instant::now();
    let prompt_tokens = tokenizer.encode(&prompt.trim().replace(' ', "▁"));
    info!("encode prompt ... {:?}", time.elapsed());

    let time = Instant::now();
    let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
    if !tokens.is_empty() {
        transformer.update(tokens, &mut kv_cache, 0);
    }
    info!("prefill transformer ... {:?}", time.elapsed());

    print!("{prompt}");

    let mut token = *last;
    let mut pos = tokens.len();
    let time = Instant::now();
    while pos < step {
        let logits = transformer.forward(token, &mut kv_cache, pos as _);
        let next = argmax(&logits);

        token = next;
        pos += 1;

        print!("{}", tokenizer.decode(next).replace('▁', " "));
        std::io::stdout().flush().unwrap();
    }
    println!();
    let duration = time.elapsed();
    info!("generate ... {duration:?}");
    info!(
        "avg. speed ... {} tokens/s",
        (pos - tokens.len()) as f32 / duration.as_secs_f32()
    )
}

#[cfg(not(detected_cuda))]
fn on_nvidia_gpu(_: impl AsRef<Path>, _: impl Tokenizer, _: impl AsRef<str>, _: usize) {
    panic!("Nvidia GPU is not detected");
}

#[cfg(detected_cuda)]
fn on_nvidia_gpu(
    model_dir: impl AsRef<Path>,
    tokenizer: impl Tokenizer,
    prompt: impl AsRef<str>,
    step: usize,
) {
    let model_dir = model_dir.as_ref();
    let prompt = prompt.as_ref();

    use std::{
        fs::File,
        io::{ErrorKind::NotFound, Read},
    };
    use transformer_nvidia::{cuda, PageLockedMemory, Transformer};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        panic!("No Nvidia GPU is detected");
    };

    let time = Instant::now();

    let config = File::open(model_dir.join("config.json"));
    let config = match config {
        Ok(f) => f,
        Err(e) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let safetensors = File::open(model_dir.join("model.safetensors"));
    let mut safetensors = match safetensors {
        Ok(f) => f,
        Err(e) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };
    info!("open file {:?}", time.elapsed());

    dev.set_mempool_threshold(u64::MAX);
    dev.context().apply(|ctx| {
        let time = Instant::now();
        let mut host = PageLockedMemory::new(safetensors.metadata().unwrap().len() as _, ctx);
        safetensors.read_exact(&mut host).unwrap();
        drop(safetensors);
        info!("read to host {:?}", time.elapsed());

        let cpy = ctx.stream();

        let time = Instant::now();
        let host = Memory::load_safetensors(config, host, false).unwrap();
        let transformer = Transformer::new(&host, &cpy);
        info!("build model host: {:?}", time.elapsed());

        let time = Instant::now();
        let prompt_tokens = tokenizer.encode(&prompt.trim().replace(' ', "▁"));
        info!("encode prompt ... {:?}", time.elapsed());

        let time = Instant::now();
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
        if !tokens.is_empty() {
            transformer.update(tokens, 0, &ctx.stream());
        }
        info!("prefill transformer ... {:?}", time.elapsed());

        print!("{prompt}");

        let mut _token = *last;
        let mut pos = tokens.len();
        let time = Instant::now();
        while pos < step {
            // let logits = transformer.forward(token, &mut kv_cache, pos as _);
            // let next = argmax(&logits);

            // token = next;
            pos += 1;

            // print!("{}", tokenizer.decode(next).replace('▁', " "));
            std::io::stdout().flush().unwrap();
        }
        println!();
        let duration = time.elapsed();
        info!("generate ... {duration:?}");
        info!(
            "avg. speed ... {} tokens/s",
            (pos - tokens.len()) as f32 / duration.as_secs_f32()
        )
    });
}

fn argmax<T: PartialOrd>(logits: &[T]) -> utok {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as _
}
