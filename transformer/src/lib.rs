//! Common code for transformers.

#![deny(warnings)]

mod blas;
mod buffer;
mod cache;
mod parameters;
mod pos;
mod request;
mod sample;

pub use blas::Matrix;
pub use buffer::LayerBuffer;
pub use cache::LayerCache;
pub use parameters::{save, Llama2, Memory, SafeTensorError};
pub use pos::pos;
pub use request::Request;
pub use sample::{BetweenF32, Sample, SampleArgs};

pub trait Transformer {
    type Cache;

    fn model(&self) -> &dyn Llama2;
    fn new_cache(&self) -> Vec<LayerCache<Self::Cache>>;
    fn decode<Id>(
        &self,
        requests: Vec<Request<Id, Self::Cache>>,
        sample: &SampleArgs,
    ) -> Vec<(Id, common::utok)>;
}
