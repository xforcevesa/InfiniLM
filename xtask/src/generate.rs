use common::utok;
use log::LevelFilter;
use simple_logger::SimpleLogger;
use std::{
    alloc::Layout,
    collections::HashMap,
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Mutex,
    time::Instant,
};
use tokenizer::{Tokenizer, VocabTxt, BPE};
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
    /// Tokenizer file.
    #[clap(short, long)]
    tokenizer: Option<String>,
    /// Max steps.
    #[clap(short, long)]
    step: Option<usize>,
    /// Copy model parameters inside memory.
    #[clap(long)]
    inside_mem: bool,
    /// Add bos before first token.
    #[clap(long)]
    insert_bos: bool,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
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
        let tokenizer = tokenizer(self.tokenizer, &model_dir);
        info!("build tokenizer ... {:?}", time.elapsed());

        let mut prompt = String::new();
        if self.insert_bos {
            prompt.push_str("<s>");
        }
        match self.prompt.chars().next() {
            Some(c) if c.is_ascii_alphabetic() => prompt.push(' '),
            _ => {}
        }
        prompt.push_str(&self.prompt);

        if self.nvidia {
            let preload_layers = if self.inside_mem { usize::MAX } else { 3 };
            on_nvidia_gpu(model_dir, tokenizer, prompt, step, preload_layers)
        } else {
            on_host(model_dir, tokenizer, prompt, step, self.inside_mem)
        }
    }
}

fn tokenizer(path: Option<String>, model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer> {
    match path {
        Some(path) => match Path::new(&path).extension() {
            Some(ext) if ext == "txt" => Box::new(VocabTxt::from_txt_file(path).unwrap()),
            Some(ext) if ext == "model" => Box::new(BPE::from_model_file(path).unwrap()),
            _ => panic!("Tokenizer file {path:?} not supported"),
        },
        None => {
            match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
                Ok(bpe) => return Box::new(bpe),
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => panic!("{e:?}"),
            }
            match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
                Ok(voc) => return Box::new(voc),
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => panic!("{e:?}"),
            }
            panic!("Tokenizer file not found");
        }
    }
}

fn on_host(
    model_dir: impl AsRef<Path>,
    tokenizer: Box<dyn Tokenizer>,
    prompt: impl AsRef<str>,
    step: usize,
    inside_mem: bool,
) {
    let model_dir = model_dir.as_ref();
    let prompt = prompt.as_ref();

    let time = Instant::now();
    let mut model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
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
        let next = argmax(logits);

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
fn on_nvidia_gpu(
    _: impl AsRef<Path>,
    _: Box<dyn Tokenizer>,
    _: impl AsRef<str>,
    _: usize,
    _: usize,
) {
    panic!("Nvidia GPU is not detected");
}

#[cfg(detected_cuda)]
fn on_nvidia_gpu(
    model_dir: impl AsRef<Path>,
    tokenizer: Box<dyn Tokenizer>,
    prompt: impl AsRef<str>,
    step: usize,
    preload_layers: usize,
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

        let compute = ctx.stream();
        let transfer = ctx.stream();

        let time = Instant::now();
        let host = Memory::load_safetensors(config, host, false).unwrap();
        let mut transformer = Transformer::new(&host, preload_layers, &transfer);
        let kv_cache = transformer.new_cache(&compute);
        info!("build model host: {:?}", time.elapsed());

        let step = step.min(host.max_position_embeddings());
        let time = Instant::now();
        let prompt_tokens = tokenizer.encode(&prompt.trim().replace(' ', "▁"));
        info!("encode prompt ... {:?}", time.elapsed());

        let time = Instant::now();
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
        if !tokens.is_empty() {
            transformer.update(tokens, &kv_cache, 0, &compute, &transfer);
        }
        info!("prefill transformer ... {:?}", time.elapsed());

        print!("{prompt}");

        let mut token = *last;
        let mut pos = tokens.len();
        let time = Instant::now();
        while pos < step {
            let logits = transformer.forward(token, &kv_cache, pos as _, &compute, &transfer);
            let next = argmax(logits);

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
