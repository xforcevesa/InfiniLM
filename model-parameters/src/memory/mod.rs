mod cast;
mod realloc;
mod safe_tensors;

use crate::{ConfigJson, DataType, Llama2, Storage};
use common::utok;
use tensor::{udim, Shape, Tensor};

pub use realloc::Allocator;
pub use safe_tensors::SafeTensorError;
pub(crate) use safe_tensors::SafeTensorHeaderJson;

pub struct Memory {
    config: ConfigJson,
    embed_tokens: Tensor<Storage>,
    layers: Vec<Layer>,
    model_norm: Tensor<Storage>,
    lm_head: Tensor<Storage>,
}

struct Layer {
    input_layernorm: Tensor<Storage>,
    w_qkv: Tensor<Storage>,
    self_attn_o_proj: Tensor<Storage>,
    post_attention_layernorm: Tensor<Storage>,
    mlp_gate_up: Tensor<Storage>,
    mlp_down: Tensor<Storage>,
}

impl Llama2 for Memory {
    #[inline]
    fn bos_token_id(&self) -> utok {
        self.config.bos_token_id
    }

    #[inline]
    fn eos_token_id(&self) -> utok {
        self.config.eos_token_id
    }

    #[inline]
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[inline]
    fn intermediate_size(&self) -> usize {
        self.config.intermediate_size
    }

    #[inline]
    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    #[inline]
    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    #[inline]
    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    #[inline]
    fn num_key_value_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    #[inline]
    fn rms_norm_eps(&self) -> f32 {
        self.config.rms_norm_eps
    }

    #[inline]
    fn rope_theta(&self) -> f32 {
        self.config.rope_theta
    }

    #[inline]
    fn data_type(&self) -> DataType {
        self.config.torch_dtype
    }

    #[inline]
    fn embed_tokens(&self) -> Tensor<Storage> {
        self.embed_tokens.clone()
    }

    #[inline]
    fn input_layernorm(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].input_layernorm.clone()
    }

    #[inline]
    fn w_qkv(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].w_qkv.clone()
    }

    #[inline]
    fn self_attn_q_proj(&self, layer: usize) -> Tensor<Storage> {
        let d = self.config.hidden_size;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.end = physical.range.start + d * d * dt;
        Tensor::new(self.config.torch_dtype, &[d as _, d as _], physical)
    }

    #[inline]
    fn self_attn_k_proj(&self, layer: usize) -> Tensor<Storage> {
        let d = self.config.hidden_size;
        let dkv = self.kv_hidden_size();
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.start += d * d * dt;
        physical.range.end = physical.range.start + dkv * d * dt;
        Tensor::new(self.config.torch_dtype, &[dkv as _, d as _], physical)
    }

    #[inline]
    fn self_attn_v_proj(&self, layer: usize) -> Tensor<Storage> {
        let d = self.config.hidden_size;
        let dkv = self.kv_hidden_size();
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.start += (d + dkv) * d * dt;
        physical.range.end = physical.range.start + dkv * d * dt;
        Tensor::new(self.config.torch_dtype, &[dkv as _, d as _], physical)
    }

    #[inline]
    fn self_attn_o_proj(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].self_attn_o_proj.clone()
    }

    #[inline]
    fn post_attention_layernorm(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].post_attention_layernorm.clone()
    }

    #[inline]
    fn mlp_gate_up(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].mlp_gate_up.clone()
    }

    #[inline]
    fn mlp_gate(&self, layer: usize) -> Tensor<Storage> {
        let di = self.config.intermediate_size;
        let d = self.config.hidden_size;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].mlp_gate_up.physical().clone();
        physical.range.end = physical.range.start + di * d * dt;
        Tensor::new(self.config.torch_dtype, &[di as _, d as _], physical)
    }

    #[inline]
    fn mlp_down(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].mlp_down.clone()
    }

    #[inline]
    fn mlp_up(&self, layer: usize) -> Tensor<Storage> {
        let di = self.config.intermediate_size;
        let d = self.config.hidden_size;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].mlp_gate_up.physical().clone();
        physical.range.start += di * d * dt;
        physical.range.end = physical.range.start + di * d * dt;
        Tensor::new(self.config.torch_dtype, &[di as _, d as _], physical)
    }

    #[inline]
    fn model_norm(&self) -> Tensor<Storage> {
        self.model_norm.clone()
    }

    #[inline]
    fn lm_head(&self) -> Tensor<Storage> {
        self.lm_head.clone()
    }
}

fn concat0(tensors: &[&Tensor<Storage>]) -> Tensor<Storage> {
    assert!(!tensors.is_empty());
    let data_type = tensors[0].data_type();
    let len = tensors[0].shape()[1..].iter().product::<udim>();

    assert!({
        tensors
            .iter()
            .skip(1)
            .all(|t| t.data_type() == data_type && t.shape()[1..].iter().product::<udim>() == len)
    });

    let shape = Shape::from_slice(&[tensors.iter().map(|t| t.shape()[0]).sum(), len]);
    let mut data = vec![0u8; shape.iter().product::<udim>() as usize * data_type.size()];
    let mut offset = 0;
    for t in tensors {
        let len = t.bytes_size();
        unsafe { t.reform_to_raw(&mut data[offset..][..len]) };
        offset += len;
    }

    Tensor::new(data_type, &shape, Storage::from_blob(data))
}

#[test]
fn test_load() {
    use std::{io::ErrorKind::NotFound, time::Instant};

    let t0 = Instant::now();
    let safetensors = Memory::load_safetensors_from_dir("../../TinyLlama-1.1B-Chat-v1.0");
    let t1 = Instant::now();
    println!("mmap {:?}", t1 - t0);

    let safetensors = match safetensors {
        Ok(m) => m,
        Err(SafeTensorError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let t0 = Instant::now();
    let _inside_memory = Memory::cast(&safetensors, DataType::F32);
    let t1 = Instant::now();
    println!("cast {:?}", t1 - t0);
}
