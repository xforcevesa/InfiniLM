mod inside_memory;
mod safe_tensors;

use crate::{ConfigJson, DataType, LayerParamsOffset, Llama2};
use common::utok;
pub use safe_tensors::SafeTensorError;

pub struct Memory<T> {
    config: ConfigJson,
    blob: T,
    embed_tokens: usize,
    layers: Vec<LayerParamsOffset>,
    model_norm: usize,
    lm_head: usize,
}

impl<T: AsRef<[u8]>> Llama2 for Memory<T> {
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
    fn embed_tokens(&self) -> &[u8] {
        let d = self.config.hidden_size;
        let dv = self.config.vocab_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.embed_tokens..][..dv * d * dt]
    }

    #[inline]
    fn input_layernorm(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].input_layernorm..][..d * dt]
    }

    #[inline]
    fn self_attn_q_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].self_attn_q_proj..][..d * d * dt]
    }

    #[inline]
    fn self_attn_k_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dkv = d * self.config.num_key_value_heads / self.config.num_attention_heads;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].self_attn_k_proj..][..dkv * d * dt]
    }

    #[inline]
    fn self_attn_v_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dkv = d * self.config.num_key_value_heads / self.config.num_attention_heads;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].self_attn_v_proj..][..dkv * d * dt]
    }

    #[inline]
    fn self_attn_o_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].self_attn_o_proj..][..d * d * dt]
    }

    #[inline]
    fn post_attention_layernorm(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].post_attention_layernorm..][..d * dt]
    }

    #[inline]
    fn mlp_gate(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let di = self.config.intermediate_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].mlp_gate..][..di * d * dt]
    }

    #[inline]
    fn mlp_down(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let di = self.config.intermediate_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].mlp_down..][..d * di * dt]
    }

    #[inline]
    fn mlp_up(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let di = self.config.intermediate_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.layers[layer].mlp_up..][..di * d * dt]
    }

    #[inline]
    fn model_norm(&self) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.model_norm..][..d * dt]
    }

    #[inline]
    fn lm_head(&self) -> &[u8] {
        let d = self.config.hidden_size;
        let dv: usize = self.config.vocab_size;
        let dt: usize = self.data_type().size();
        &self.blob.as_ref()[self.lm_head..][..dv * d * dt]
    }
}

#[test]
fn test_load() {
    use std::time::Instant;

    // set env for POWERSHELL: `$env:RUST_LOG="INFO";`
    env_logger::init();

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
