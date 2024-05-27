use common::{utok, FileLoadError};
use std::{fs, path::Path};
use tensor::DataType;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct ConfigJson {
    pub bos_token_id: utok,
    pub eos_token_id: utok,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: DataType,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

impl ConfigJson {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        let path = model_dir.as_ref().join("config.json");
        let content = fs::read_to_string(path).map_err(FileLoadError::Io)?;
        serde_json::from_str(&content).map_err(FileLoadError::Json)
    }
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}
