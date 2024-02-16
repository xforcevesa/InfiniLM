mod data_type;
mod memory;
mod save;

#[macro_use]
extern crate log;

use std::fmt;

use common::utok;

pub use data_type::DataType;
pub use save::save;

pub trait Llama2 {
    fn bos_token_id(&self) -> utok;
    fn eos_token_id(&self) -> utok;
    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f32;
    fn rope_theta(&self) -> f32;
    fn data_type(&self) -> DataType;

    fn size(&self) -> usize {
        let d = self.hidden_size();
        let dv = self.vocab_size();
        let dkv = d * self.num_key_value_heads() / self.num_attention_heads();
        let di = self.intermediate_size();
        let l = self.num_hidden_layers();

        (d * dv      // embed_tokens
       + l * d       // input_layernorm
       + l * d * d   // self_attn_q_proj
       + l * dkv * d // self_attn_k_proj
       + l * dkv * d // self_attn_v_proj
       + l * d * d   // self_attn_o_proj
       + l * d       // post_attention_layernorm
       + l * di * d  // mlp_gate
       + l * d * di  // mlp_down
       + l * di * d  // mlp_up
       + d           // model_norm
       + dv * d)     // lm_head
       * self.data_type().size()
    }

    fn embed_tokens(&self) -> &[u8];
    fn input_layernorm(&self, layer: usize) -> &[u8];
    fn self_attn_q_proj(&self, layer: usize) -> &[u8];
    fn self_attn_k_proj(&self, layer: usize) -> &[u8];
    fn self_attn_v_proj(&self, layer: usize) -> &[u8];
    fn self_attn_o_proj(&self, layer: usize) -> &[u8];
    fn post_attention_layernorm(&self, layer: usize) -> &[u8];
    fn mlp_gate(&self, layer: usize) -> &[u8];
    fn mlp_down(&self, layer: usize) -> &[u8];
    fn mlp_up(&self, layer: usize) -> &[u8];
    fn model_norm(&self) -> &[u8];
    fn lm_head(&self) -> &[u8];
}

pub use memory::{Memory, SafeTensorError};
use serde::Serialize;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct ConfigJson {
    pub bos_token_id: utok,
    pub eos_token_id: utok,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(serialize_with = "serialize_float")]
    pub rms_norm_eps: f32,
    #[serde(serialize_with = "serialize_float")]
    pub rope_theta: f32,
    pub torch_dtype: DataType,
}

fn serialize_float<S>(val: &impl fmt::LowerExp, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    format!("{val:e}").serialize(s)
}

struct LayerParamsOffset {
    input_layernorm: usize,
    self_attn_q_proj: usize,
    self_attn_k_proj: usize,
    self_attn_v_proj: usize,
    self_attn_o_proj: usize,
    post_attention_layernorm: usize,
    mlp_gate: usize,
    mlp_down: usize,
    mlp_up: usize,
}