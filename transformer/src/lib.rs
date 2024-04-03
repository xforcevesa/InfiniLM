//! Common code for transformers.

#![deny(warnings)]

mod blas;
mod buffer;
mod cache;
mod kernels;
mod parameters;
mod pos;
mod request;
mod sample;

pub use blas::Matrix;
pub use buffer::LayerBuffer;
pub use cache::LayerCache;
pub use kernels::Kernels;
pub use parameters::{save, DistributedLayer, Distributer, Llama2, Memory, SafeTensorError};
pub use pos::pos;
pub use request::Request;
pub use sample::{BetweenF32, SampleArgs};

use common::utok;
use tensor::Tensor;

pub trait Transformer {
    type Cache;

    fn model(&self) -> &dyn Llama2;
    fn new_cache(&self) -> Vec<LayerCache<Self::Cache>>;
    fn decode<Id>(&self, requests: Vec<Request<Id, Self::Cache>>)
        -> (Vec<Id>, Tensor<Self::Cache>);
    fn sample<Id>(
        &self,
        args: &SampleArgs,
        requests: Vec<Id>,
        logits: Tensor<Self::Cache>,
    ) -> Vec<(Id, utok)>;
}
