use crate::layout;
use operators::{
    common_cpu::ThisThread,
    mat_mul::{
        common_cpu::{Operator as MatMul, Scheme as MatMulScheme},
        LayoutAttrs,
    },
    Operator, Scheme, F16,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

/// c = a x b
pub fn mat_mul<T, U, V>(c: &mut Tensor<T>, beta: f32, a: &Tensor<U>, b: &Tensor<V>, alpha: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    MatMulScheme::new(
        &MatMul::new(&F16).unwrap(),
        LayoutAttrs {
            c: layout(c),
            a: layout(a),
            b: layout(b),
        },
    )
    .unwrap()
    .launch(
        &(
            c.physical_mut().as_mut_ptr(),
            beta,
            a.physical().as_ptr(),
            b.physical().as_ptr(),
            alpha,
        ),
        &ThisThread,
    );
}
