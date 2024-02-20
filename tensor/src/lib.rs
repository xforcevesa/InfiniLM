mod data_type;
mod fmt;
mod operator;
mod tensor;

#[allow(non_camel_case_types)]
pub type udim = u32;

#[allow(non_camel_case_types)]
pub type idim = i32;

pub use data_type::DataType;
pub use operator::{Broadcast, Operator, Slice, SliceDim, Split, Squeeze, SqueezeOp, Transpose};
pub use tensor::{Affine, Shape, Tensor};
