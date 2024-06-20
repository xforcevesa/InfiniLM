#![cfg(detected_neuware)]

mod resource;

use causal_lm::{CausalLM, Model};
use common::FileLoadError;
use common_cn::Tensor;

pub use common_cn::{cndrv, synchronize};
pub use resource::Cache;

pub struct Transformer;

impl Model for Transformer {
    type Meta = ();
    type Error = FileLoadError;

    fn load(
        _model_dir: impl AsRef<std::path::Path>,
        _meta: Self::Meta,
    ) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    fn max_seq_len(&self) -> common::upos {
        todo!()
    }

    fn eos_token(&self) -> common::utok {
        todo!()
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        todo!()
    }

    fn duplicate_cache(
        &self,
        _cache: &Tensor<Self::Storage>,
        _pos: common::upos,
    ) -> Tensor<Self::Storage> {
        todo!()
    }

    fn token_embed(
        &self,
        _queries: impl IntoIterator<Item = common::utok>,
    ) -> Tensor<Self::Storage> {
        todo!()
    }

    fn forward<'a>(
        &self,
        _queries: impl IntoIterator<Item = causal_lm::QueryContext<'a, Self::Storage>>,
        _token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        todo!()
    }

    fn decode(
        &self,
        _decoding: impl IntoIterator<Item = causal_lm::DecodingMeta>,
        _hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        todo!()
    }

    fn sample(
        &self,
        _args: impl IntoIterator<Item = causal_lm::SampleMeta>,
        _logits: Tensor<Self::Storage>,
    ) -> Vec<common::utok> {
        todo!()
    }
}
