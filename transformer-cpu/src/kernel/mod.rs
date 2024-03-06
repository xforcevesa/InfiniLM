mod fused_softmax;
mod gather;
mod mat_mul;
mod rms_norm;
mod rotary_embedding;
mod swiglu;

pub(super) use fused_softmax::softmax;
pub(super) use gather::gather;
pub(super) use mat_mul::mat_mul;
pub(super) use rms_norm::rms_norm;
pub(super) use rotary_embedding::rotary_embedding;
pub(super) use swiglu::swiglu;

macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width..][..$width]
    };
}

use slice;
