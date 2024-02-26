use common::utok;
use log::LevelFilter;
use simple_logger::SimpleLogger;
use std::{io::Write, path::PathBuf, time::Instant};
use tokenizer::{Tokenizer, BPE};
use transformer_cpu::{model_parameters::Memory, Transformer};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Prompt.
    #[clap(short, long)]
    prompt: String,
    /// Log level.
    #[clap(short, long)]
    log: Option<String>,
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
        let model = Box::new(Memory::load_safetensors(&model_dir).unwrap());
        info!("load model ... {:?}", time.elapsed());

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
        loop {
            let logits = transformer.forward(token, &mut kv_cache, pos as _);
            let next = argmax(&logits);

            token = next;
            pos += 1;

            print!("{}", tokenizer.decode(next).replace('▁', " "));
            std::io::stdout().flush().unwrap();
        }
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
