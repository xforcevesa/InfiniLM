mod cast;
mod json;
mod load;
mod save;

use common::{safe_tensors::SharedTensor, utok, Blob};
use std::{ops::Deref, sync::Arc};
use tensor::{udim, DataType, Tensor};

pub struct Storage {
    pub config: InferenceConfig,

    pub embed_tokens: Tensor<Weight>,
    pub layers: Vec<LayerStorage>,
    pub lm_layernorm: Tensor<Weight>,
    pub lm_head: Tensor<Weight>,
}

pub struct LayerStorage {
    pub att_layernorm: Tensor<Weight>,
    pub att_qkv: Tensor<Weight>,
    pub att_o: Tensor<Weight>,
    pub mlp_layernorm: Tensor<Weight>,
    pub mlp_gate_up: Tensor<Weight>,
    pub mlp_down: Tensor<Weight>,
}

#[derive(Clone, Debug)]
pub struct InferenceConfig {
    pub dt: DataType,
    pub voc: udim,
    pub nlayers: udim,
    pub nh: udim,
    pub nkvh: udim,
    pub d: udim,
    pub dkv: udim,
    pub di: udim,
    pub max_seq_len: udim,
    pub bos_token: utok,
    pub eos_token: utok,
    pub epsilon: f32,
    pub theta: f32,
}

#[derive(Clone)]
pub enum Weight {
    SafeTensor(SharedTensor),
    Blob(Arc<Blob>),
}

impl From<SharedTensor> for Weight {
    #[inline]
    fn from(tensor: SharedTensor) -> Self {
        Self::SafeTensor(tensor)
    }
}

impl From<Blob> for Weight {
    #[inline]
    fn from(blob: Blob) -> Self {
        Self::Blob(Arc::new(blob))
    }
}

impl Deref for Weight {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            Self::SafeTensor(tensor) => tensor,
            Self::Blob(blob) => blob,
        }
    }
}
