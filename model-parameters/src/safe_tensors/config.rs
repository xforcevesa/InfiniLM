use common::utok;
use safetensors::tensor::TensorInfo;
use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct ConfigJson {
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
    pub torch_dtype: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct SafeTensorHeaderJson {
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(rename = "__metadata__")]
    pub meta: Option<HashMap<String, serde_json::Value>>,
}
