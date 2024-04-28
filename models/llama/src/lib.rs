use common::{safe_tensors::Dtype, utok, FileLoadError};
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

pub fn convert(dtype: Dtype) -> DataType {
    match dtype {
        Dtype::BOOL => DataType::Bool,
        Dtype::I8 => DataType::I8,
        Dtype::I16 => DataType::I16,
        Dtype::I32 => DataType::I32,
        Dtype::I64 => DataType::I64,
        Dtype::U8 => DataType::U8,
        Dtype::U16 => DataType::U16,
        Dtype::U32 => DataType::U32,
        Dtype::U64 => DataType::U64,
        Dtype::F16 => DataType::F16,
        Dtype::BF16 => DataType::BF16,
        Dtype::F32 => DataType::F32,
        Dtype::F64 => DataType::F64,
        _ => unreachable!(),
    }
}
