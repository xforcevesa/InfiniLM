mod cache;

use cache::LayerCache;
use common::{upos, utok};
use model_parameters::{DataType, Llama2, Memory};

pub extern crate model_parameters;

pub struct Transformer {
    model: Box<dyn Llama2>,
}

impl Transformer {
    pub fn new(model: Box<dyn Llama2>) -> Self {
        let model = match model.data_type() {
            DataType::BF16 => Box::new(Memory::cast(&*model, DataType::F32)),
            _ => model,
        };
        Self { model }
    }

    #[inline]
    pub fn new_cache(&self) -> Vec<LayerCache> {
        LayerCache::new_layers(&*self.model)
    }

    pub fn update(
        &self,
        _tokens: &[utok],
        _cache: Option<&mut [LayerCache]>,
        _pos: upos,
    ) -> Vec<f32> {
        vec![]
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
    let _transformer = Transformer::new(Box::new(safetensors));
    let t1 = Instant::now();
    println!("build transformer {:?}", t1 - t0);
}
