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
pub use rms_norm::rms_norm;
pub use rotary_embedding::rotary_embedding;
pub use swiglu::swiglu;
