mod safe_tensors;

pub enum DataType {
    F16,
    BF16,
    F32,
}

impl DataType {
    #[inline]
    pub const fn size(&self) -> usize {
        match self {
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::F32 => 4,
        }
    }
}

pub trait LLama2 {
    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn data_type(&self) -> DataType;

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

pub use safe_tensors::SafeTensors;
