#![cfg(detected_nccl)]

mod parameters;

#[macro_use]
extern crate log;

pub use common_nv::cuda;

use parameters::ParameterMatrix;

pub struct Transformer {
    comms: nccl::CommunicatorGroup,
    matrix: ParameterMatrix,
}

impl Transformer {}
