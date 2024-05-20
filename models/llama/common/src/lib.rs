mod cast;
mod compute;
mod json;
mod load;
mod save;

use common::{safe_tensors::SharedTensor, upos, utok, Blob};
use std::{ops::Deref, sync::Arc};
use tensor::{slice, udim, DataType, Tensor};

pub use compute::{ComputeStream, LLamaLayer};

pub struct Storage {
    pub config: InferenceConfig,

    pub embed_tokens: Tensor<Weight>,
    pub layers: Vec<LayerStorage<Weight>>,
    pub lm_layernorm: Tensor<Weight>,
    pub lm_head: Tensor<Weight>,
}

pub struct LayerStorage<T> {
    pub att_layernorm: Tensor<T>,
    pub att_qkv: Tensor<T>,
    pub att_o: Tensor<T>,
    pub mlp_layernorm: Tensor<T>,
    pub mlp_gate_up: Tensor<T>,
    pub mlp_down: Tensor<T>,
}

impl<T> LayerStorage<T> {
    pub fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> LayerStorage<U> {
        macro_rules! map {
            ($($ident:ident)+) => {
                LayerStorage {$(
                    $ident: self.$ident.as_ref().map_physical(&mut f),
                )+}
            };
        }
        map! {
            att_layernorm
            att_qkv
            att_o
            mlp_layernorm
            mlp_gate_up
            mlp_down
        }
    }
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

impl InferenceConfig {
    pub fn new_cache<S>(&self, f: impl FnOnce(usize) -> S) -> Tensor<S> {
        Tensor::alloc(
            self.dt,
            &[
                self.nlayers,
                2,
                self.nkvh,
                self.max_seq_len,
                self.d / self.nh,
            ],
            f,
        )
    }

    pub fn duplicate_cache<S>(
        &self,
        cache: &Tensor<S>,
        pos: upos,
        malloc: impl FnOnce(usize) -> S,
        reform: impl FnOnce(Tensor<&mut S>, Tensor<&S>),
    ) -> Tensor<S> {
        let mut ans = Tensor::alloc(cache.data_type(), cache.shape(), malloc);
        if pos > 0 {
            let &[_nlayers, 2, _nkvh, max_seq_len, _dh] = cache.shape() else {
                panic!()
            };
            assert!(pos <= max_seq_len);
            let slice = [
                slice![=>],
                slice![=>],
                slice![=>],
                slice![=>pos],
                slice![=>],
            ];
            reform(ans.as_mut().slice(&slice), cache.as_ref().slice(&slice));
        }
        ans
    }
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
