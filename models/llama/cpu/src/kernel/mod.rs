macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

pub(super) use slice;
pub(super) mod fused_softmax;
pub(super) mod gather;
pub(super) mod mat_mul;
pub(super) mod rms_norm;
pub(super) mod rotary_embedding;
pub(super) mod swiglu;
