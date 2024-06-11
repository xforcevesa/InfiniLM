use crate::layout;
use operators::{
    common_cpu::ThisThread,
    rms_norm::{
        common_cpu::{Operator as RmsNorm, Scheme as RmsNormScheme},
        LayoutAttrs,
    },
    Operator, Scheme, F16,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn rms_norm<T, U, V>(o: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>, epsilon: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    RmsNormScheme::new(
        &RmsNorm::new(&F16).unwrap(),
        LayoutAttrs {
            y: layout(o),
            x: layout(x),
            w: layout(w),
        },
    )
    .unwrap()
    .launch(
        &(
            o.physical_mut().as_mut_ptr(),
            x.physical().as_ptr(),
            w.physical().as_ptr(),
            epsilon,
        ),
        &ThisThread,
    );
}
