use std::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use operators::{
    rms_norm::{self, RmsNorm},
    Device, QueueOf, TensorLayout, F16, U32,
};
use tensor::{DataType, Tensor};

pub fn layout<T>(t: &Tensor<T>) -> TensorLayout {
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

pub fn rms_norm<S, D, Y, X, W>(
    _: PhantomData<S>,
    op: &S::Operator,
    y: &mut Tensor<Y>,
    x: &Tensor<X>,
    w: &Tensor<W>,
    epsilon: f32,
    queue: &QueueOf<D>,
) where
    D: Device,
    S: RmsNorm<D>,
    S::Error: fmt::Debug,
    Y: DerefMut<Target = [D::Byte]>,
    X: Deref<Target = [D::Byte]>,
    W: Deref<Target = [D::Byte]>,
{
    S::new(
        op,
        rms_norm::LayoutAttrs {
            y: layout(y),
            x: layout(x),
            w: layout(w),
        },
    )
    .unwrap()
    .launch(
        &(
            y.physical_mut().as_mut_ptr(),
            x.physical().as_ptr(),
            w.physical().as_ptr(),
            epsilon,
        ),
        queue,
    );
}
