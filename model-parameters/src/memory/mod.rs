mod cast;
mod safe_tensors;

use crate::{ConfigJson, DataType, Llama2, Storage};
use common::utok;
use tensor::{Shape, Tensor};

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
    mlp_gate: Tensor<Storage>,
    mlp_down: Tensor<Storage>,
    mlp_up: Tensor<Storage>,
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
        Tensor::new(self.config.torch_dtype, &[d, d], physical)
    }

    #[inline]
    fn self_attn_k_proj(&self, layer: usize) -> Tensor<Storage> {
        let d = self.config.hidden_size;
        let dkv = d * self.config.num_key_value_heads / self.config.num_attention_heads;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.start += d * d * dt;
        physical.range.end = physical.range.start + dkv * d * dt;
        Tensor::new(self.config.torch_dtype, &[dkv, d], physical)
    }

    #[inline]
    fn self_attn_v_proj(&self, layer: usize) -> Tensor<Storage> {
        let d = self.config.hidden_size;
        let dkv = d * self.config.num_key_value_heads / self.config.num_attention_heads;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.start += (d + dkv) * d * dt;
        physical.range.end = physical.range.start + dkv * d * dt;
        Tensor::new(self.config.torch_dtype, &[dkv, d], physical)
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
    fn mlp_gate(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].mlp_gate.clone()
    }

    #[inline]
    fn mlp_down(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].mlp_down.clone()
    }

    #[inline]
    fn mlp_up(&self, layer: usize) -> Tensor<Storage> {
        self.layers[layer].mlp_up.clone()
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
    let mut shape = Shape::from_slice(tensors[0].shape());

    debug_assert!(tensors
        .iter()
        .all(|t| t.data_type() == data_type && t.shape()[1..] == shape[1..]));

    for t in &tensors[1..] {
        shape[0] += t.shape()[0];
    }

    let size = shape.iter().map(|&d| d as usize).product::<usize>() * data_type.size();
    let mut data = vec![0u8; size];
    let mut offset = 0;
    for t in tensors {
        let len = t.size() * data_type.size();
        data[offset..][..len].copy_from_slice(t.as_slice());
        offset += len;
    }

    let shape = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
    Tensor::new(data_type, &shape, Storage::from_blob(data))
}

#[test]
fn test_load() {
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
    let _inside_memory = Memory::cast(&safetensors, DataType::F32);
    let t1 = Instant::now();
    println!("cast {:?}", t1 - t0);
}
