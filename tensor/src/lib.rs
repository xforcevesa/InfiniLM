mod data_type;
mod fmt;
mod operator;
mod pattern;
mod tensor;

#[allow(non_camel_case_types)]
pub type udim = u32;

#[allow(non_camel_case_types)]
pub type idim = i32;

pub use data_type::DataType;
pub use operator::{Operator, SliceDim};
pub use pattern::{expand_indices, idx_strides, Affine, Shape};
pub use tensor::{Storage, Tensor};
