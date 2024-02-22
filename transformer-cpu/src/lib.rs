mod cache;
mod kernel;
mod storage;

use cache::LayerCache;
use common::{upos, utok};
use kernel::{gather, matmul, rms_norm, rotary_embedding};
use model_parameters::{Llama2, Memory};
use storage::Storage;
use tensor::{udim, DataType, Tensor};

pub extern crate model_parameters;

pub struct Transformer {
    model: Box<dyn Llama2>,
}

impl Transformer {
    #[inline]
    pub fn new(model: Box<dyn Llama2>) -> Self {
        Self {
            model: match model.data_type() {
                DataType::BF16 => Box::new(Memory::cast(&*model, DataType::F32)),
                _ => model,
            },
        }
    }

    #[inline]
    pub fn new_cache(&self) -> Vec<LayerCache> {
        LayerCache::new_layers(&*self.model)
    }

    pub fn update(
        &self,
        tokens: &[utok],
        _cache: Option<&mut [LayerCache]>,
        pos: upos,
    ) -> Vec<f32> {
        let seq_len = tokens.len() as udim;
        let d = self.model.hidden_size() as udim;
        let dkv = self.model.kv_hidden_size() as udim;
        let dh = d / self.model.num_attention_heads() as udim;
        let dt = self.model.data_type();

        #[inline]
        fn tensor(dt: DataType, shape: &[udim]) -> Tensor<Storage> {
            Tensor::new(
                dt,
                shape,
                Storage::new(shape.iter().product::<udim>() as usize * dt.size()),
            )
        }
        // println!("tokens: {tokens:?}");

        let mut a = tensor(dt, &[seq_len, d]);
        gather(&mut a.access_mut(), &self.model.embed_tokens(), tokens);
        // println!("gather: {a}");

        let mut b = tensor(dt, &[seq_len, d]);
        let mut qkv = tensor(dt, &[seq_len, d + dkv + dkv]);
        for layer in 0..self.model.num_hidden_layers() {
            // b <- rms-norm(a)
            rms_norm(
                &mut b.access_mut(),
                &a.access(),
                &self.model.input_layernorm(layer),
                self.model.rms_norm_eps(),
            );
            // println!("layer {layer} rms norm: {b}");
            // qkv = b * w_qkv
            matmul(
                &mut qkv.access_mut(),
                &b.access(),
                &self.model.w_qkv(layer).transpose(&[1, 0]),
            );
            let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
            // println!("layer {layer} q: {}", qkv[0]);
            // println!("layer {layer} k: {}", qkv[1]);
            // println!("layer {layer} v: {}", qkv[2]);
            let theta = self.model.rope_theta();
            rotary_embedding(&mut qkv[0].access_mut(), dh, pos, theta);
            rotary_embedding(&mut qkv[1].access_mut(), dh, pos, theta);
            // println!("layer {layer} rot q: {}", qkv[0].access());
            // println!("layer {layer} rot k: {}", qkv[1].access());
        }

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
