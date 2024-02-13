mod safe_tensors;

pub trait LLama2 {
    type Ptr;

    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> usize;
    fn vocab_size(&self) -> usize;

    fn embed_tokens(&self) -> Self::Ptr;
    fn input_layernorm(&self, layer: usize) -> Self::Ptr;
    fn self_attn_q_proj(&self, layer: usize) -> Self::Ptr;
    fn self_attn_k_proj(&self, layer: usize) -> Self::Ptr;
    fn self_attn_v_proj(&self, layer: usize) -> Self::Ptr;
    fn self_attn_o_proj(&self, layer: usize) -> Self::Ptr;
    fn post_attention_layernorm(&self, layer: usize) -> Self::Ptr;
    fn mlp_gate(&self, layer: usize) -> Self::Ptr;
    fn mlp_down(&self, layer: usize) -> Self::Ptr;
    fn mlp_up(&self, layer: usize) -> Self::Ptr;
    fn model_norm(&self) -> Self::Ptr;
    fn lm_head(&self) -> Self::Ptr;
}

pub use safe_tensors::SafeTensors;
