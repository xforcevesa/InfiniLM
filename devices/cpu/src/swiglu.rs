use crate::layout;
use operators::{
    common_cpu::ThisThread,
    swiglu::{
        common_cpu::{Operator as Swiglu, Scheme as SwigluScheme},
        LayoutAttrs,
    },
    Operator, Scheme, F16,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn swiglu<T, U>(gate: &mut Tensor<T>, up: &Tensor<U>)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    SwigluScheme::new(
        &Swiglu::new(&F16).unwrap(),
        LayoutAttrs {
            gate: layout(gate),
            up: layout(up),
        },
    )
    .unwrap()
    .launch(
        &(gate.physical_mut().as_mut_ptr(), up.physical().as_ptr()),
        &ThisThread,
    );
}
