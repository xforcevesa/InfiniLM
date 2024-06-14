use common_devices::layout;
use operators::{
    common_cpu::ThisThread,
    fuesd_softmax::{
        common_cpu::{Operator as FusedSoftmax, Scheme as FusedSoftmaxScheme},
        LayoutAttrs,
    },
    Operator, Scheme, F16,
};
use std::ops::DerefMut;
use tensor::Tensor;

/// - att: [nh, seq_len, att_len]
pub fn softmax<T>(att: &mut Tensor<T>)
where
    T: DerefMut<Target = [u8]>,
{
    FusedSoftmaxScheme::new(
        &FusedSoftmax::new(&F16).unwrap(),
        LayoutAttrs { att: layout(att) },
    )
    .unwrap()
    .launch(&att.physical_mut().as_mut_ptr(), &ThisThread);
}
