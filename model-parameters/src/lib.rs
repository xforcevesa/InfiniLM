mod memory;
mod save;

use common::utok;
use std::{ops::Range, sync::Arc};
use tensor::{DataType, Tensor};

pub use memory::{Memory, SafeTensorError};
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

    fn embed_tokens(&self) -> Tensor<Storage>;
    fn input_layernorm(&self, layer: usize) -> Tensor<Storage>;
    fn self_attn_q_proj(&self, layer: usize) -> Tensor<Storage>;
    fn self_attn_k_proj(&self, layer: usize) -> Tensor<Storage>;
    fn self_attn_v_proj(&self, layer: usize) -> Tensor<Storage>;
    fn self_attn_o_proj(&self, layer: usize) -> Tensor<Storage>;
    fn post_attention_layernorm(&self, layer: usize) -> Tensor<Storage>;
    fn mlp_gate(&self, layer: usize) -> Tensor<Storage>;
    fn mlp_down(&self, layer: usize) -> Tensor<Storage>;
    fn mlp_up(&self, layer: usize) -> Tensor<Storage>;
    fn model_norm(&self) -> Tensor<Storage>;
    fn lm_head(&self) -> Tensor<Storage>;
}

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
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub torch_dtype: DataType,
}

type Blob = dyn 'static + AsRef<[u8]>;

#[derive(Clone)]
pub struct Storage {
    data: Arc<Blob>,
    range: Range<usize>,
}

impl Storage {
    #[inline]
    pub fn new(data: Arc<Blob>, offset: usize, len: usize) -> Self {
        Self {
            data,
            range: offset..offset + len,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data.as_ref().as_ref()[self.range.clone()]
    }
}
