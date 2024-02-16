﻿use std::{path::PathBuf, time::Instant};
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
}

impl GenerateArgs {
    pub fn invoke(self) {
        let model_dir = PathBuf::from(self.model);

        let time = Instant::now();
        let model = Box::new(Memory::load_safetensors(&model_dir).unwrap());
        info!("load model ... {:?}", time.elapsed());

        let time = Instant::now();
        let _transformer = Transformer::new(model, 1);
        info!("build transformer ... {:?}", time.elapsed());

        let time = Instant::now();
        let tokenizer = BPE::from_model_file(model_dir.join("tokenizer.model")).unwrap();
        info!("build tokenizer ... {:?}", time.elapsed());

        let time = Instant::now();
        let _prompt_tokens = tokenizer.encode(self.prompt.trim());
        info!("encode prompt ... {:?}", time.elapsed());
    }
}
