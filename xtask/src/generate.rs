use common::utok;
use log::LevelFilter;
use simple_logger::SimpleLogger;
use std::{
    alloc::Layout, collections::HashMap, io::Write, path::PathBuf, ptr::NonNull, sync::Mutex,
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
    #[clap(short, long)]
    inside_mem: bool,
    /// Log level.
    #[clap(short, long)]
    log: Option<String>,
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

        let time = Instant::now();
        let mut model = Box::new(Memory::load_safetensors(&model_dir).unwrap());
        info!("load model ... {:?}", time.elapsed());

        if self.inside_mem {
            let time = Instant::now();
            let allocator = NormalAllocator(Mutex::new(HashMap::new()));
            model = Box::new(Memory::realloc_with(&*model, allocator));
            info!("copy model ... {:?}", time.elapsed());
        }
        let step = self
            .step
            .unwrap_or(usize::MAX)
            .min(model.max_position_embeddings());
        let time = Instant::now();
        let mut transformer = Transformer::new(model);
        let mut kv_cache = transformer.new_cache();
        info!("build transformer ... {:?}", time.elapsed());

        let time = Instant::now();
        let tokenizer = BPE::from_model_file(model_dir.join("tokenizer.model")).unwrap();
        info!("build tokenizer ... {:?}", time.elapsed());

        let time = Instant::now();
        let prompt_tokens = tokenizer.encode(&self.prompt.trim().replace(' ', "▁"));
        info!("encode prompt ... {:?}", time.elapsed());

        let time = Instant::now();
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
        if !tokens.is_empty() {
            transformer.update(tokens, &mut kv_cache, 0);
        }
        info!("prefill transformer ... {:?}", time.elapsed());

        print!("{}", self.prompt);

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
}

fn argmax<T: PartialOrd>(logits: &[T]) -> utok {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as _
}
