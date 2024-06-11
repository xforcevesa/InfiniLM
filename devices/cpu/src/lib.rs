#[macro_export]
macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

mod fused_softmax;
mod gather;
mod mat_mul;
mod rms_norm;
mod rotary_embedding;
mod swiglu;

pub use fused_softmax::softmax;
pub use gather::gather;
pub use mat_mul::mat_mul;
use operators::{TensorLayout, F16, U32};
pub use rms_norm::rms_norm;
pub use rotary_embedding::rotary_embedding;
pub use swiglu::swiglu;
use tensor::{DataType, Tensor};

fn layout<T>(t: &Tensor<T>) -> TensorLayout {
    let dt = match t.data_type() {
        DataType::F16 => F16,
        DataType::U32 => U32,
        _ => todo!(),
    };
    let shape = t.shape().iter().map(|&x| x as usize).collect::<Vec<_>>();
    let strides = t
        .strides()
        .iter()
        .map(|&x| x as isize * t.data_type().size() as isize)
        .collect::<Vec<_>>();
    TensorLayout::new(dt, shape, strides, t.bytes_offset() as _)
}
