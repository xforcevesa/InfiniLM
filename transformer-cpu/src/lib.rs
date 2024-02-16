mod cache;

use cache::LayerCache;
use model_parameters::{DataType, Llama2, Memory};

pub struct Transformer {
    model: Box<dyn Llama2>,
    cache: Vec<LayerCache>,
}

impl Transformer {
    pub fn new(model: Box<dyn Llama2>, batch: usize) -> Self {
        let model = match model.data_type() {
            DataType::BF16 => Box::new(Memory::cast(&*model, DataType::F32)),
            _ => model,
        };
        let cache = (0..model.num_hidden_layers())
            .map(|_| LayerCache::new(&*model, batch))
            .collect();
        Self { model, cache }
    }
}

#[test]
fn test_build() {
    use model_parameters::SafeTensorError;
    use std::time::Instant;

    let t0 = Instant::now();
    let safetensors = Memory::load_safetensors("../../TinyLlama-1.1B-Chat-v1.0");
    let t1 = Instant::now();
    println!("mmap {:?}", t1 - t0);

    let safetensors = match safetensors {
        Ok(m) => m,
        Err(SafeTensorError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let t0 = Instant::now();
    let _transformer = Transformer::new(Box::new(safetensors), 1);
    let t1 = Instant::now();
    println!("build transformer {:?}", t1 - t0);
}
