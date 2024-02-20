mod cache;
mod kernel;

use cache::LayerCache;
use common::{upos, utok};
use kernel::{gather, matmul, rms_norm};
use model_parameters::{Llama2, Memory};
use tensor::{DataType, Split, Tensor, Transpose};

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
        _pos: upos,
    ) -> Vec<f32> {
        let seq_len = tokens.len();
        let d = self.model.hidden_size();
        let dkv = d * self.model.num_key_value_heads() / self.model.num_attention_heads();
        let dt = self.model.data_type();

        #[inline]
        fn tensor(dt: DataType, shape: &[usize]) -> Tensor<Vec<u8>> {
            Tensor::new(
                dt,
                shape,
                vec![0u8; shape.iter().product::<usize>() * dt.size()],
            )
        }

        println!("tokens: {tokens:?}");

        let mut a = tensor(dt, &[seq_len, d]);
        gather(&mut a, &self.model.embed_tokens(), tokens);

        println!("gather: {a}");

        let mut b = tensor(dt, &[seq_len, d]);
        let mut qkv = tensor(dt, &[d + dkv + dkv, seq_len]);
        for layer in 0..self.model.num_hidden_layers() {
            // b <- rms-norm(a)
            rms_norm(
                &mut b,
                &a,
                &self.model.input_layernorm(layer),
                self.model.rope_theta(),
            );
            println!("rms norm: {b}");
            {
                // qkv = w_qkv * b
                let w = &self.model.w_qkv(layer);
                let x = &b.apply(&Transpose::new(&[1, 0]))[0];
                matmul(&mut qkv, w, x);
            }
            let qkv = qkv.apply(&Split::new(0, &[d as _, dkv as _, dkv as _]));
            let _q = &qkv[0];
            let _k = &qkv[1];
            let _v = &qkv[2];
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
