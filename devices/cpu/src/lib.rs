#[macro_export]
macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

mod gather;

use common::utok;
use common_devices::{mat_mul, rms_norm, rope, softmax, swiglu, SliceOn};
use operators::{
    fuesd_softmax::common_cpu as softmax, mat_mul::common_cpu as mat_mul,
    rms_norm::common_cpu as rms_norm, rope::common_cpu as rope, swiglu::common_cpu as swiglu,
    Operator, QueueOf, F16,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};
use tensor::Tensor;

pub extern crate tensor;

pub use common_devices::Kernels;
pub use operators::common_cpu::{Device as Cpu, ThisThread};

pub struct CpuKernels {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    softmax: softmax::Operator,
    swiglu: swiglu::Operator,
}

impl Default for CpuKernels {
    fn default() -> Self {
        Self {
            mat_mul: mat_mul::Operator::new(&F16).unwrap(),
            rms_norm: rms_norm::Operator::new(&F16).unwrap(),
            rope: rope::Operator::new(&F16).unwrap(),
            softmax: softmax::Operator::new(&F16).unwrap(),
            swiglu: swiglu::Operator::new(&F16).unwrap(),
        }
    }
}

impl Kernels for CpuKernels {
    type Device = Cpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        _queue: &QueueOf<Self::Device>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens);
    }

    fn rms_norm<T, U, V>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<U>,
        w: &Tensor<V>,
        epsilon: f32,
        queue: &QueueOf<Self::Device>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
        V: Deref<Target = SliceOn<Self::Device>>,
    {
        rms_norm(
            PhantomData::<rms_norm::Scheme>,
            &self.rms_norm,
            y,
            x,
            w,
            epsilon,
            queue,
        );
    }

    fn rope<T, U>(
        &self,
        t: &mut Tensor<T>,
        pos: &Tensor<U>,
        theta: f32,
        queue: &QueueOf<Self::Device>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
    {
        rope(
            PhantomData::<rope::Scheme>,
            &self.rope,
            t,
            pos,
            theta,
            queue,
        );
    }

    fn mat_mul<T, U, V>(
        &self,
        c: &mut Tensor<T>,
        beta: f32,
        a: &Tensor<U>,
        b: &Tensor<V>,
        alpha: f32,
        queue: &QueueOf<Self::Device>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
        V: Deref<Target = SliceOn<Self::Device>>,
    {
        mat_mul(
            PhantomData::<mat_mul::Scheme>,
            &self.mat_mul,
            c,
            beta,
            a,
            b,
            alpha,
            queue,
        );
    }

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, _queue: &QueueOf<Self::Device>)
    where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
    {
        src.reform_to(dst);
    }

    fn softmax<T>(&self, att: &mut Tensor<T>, queue: &QueueOf<Self::Device>)
    where
        T: DerefMut<Target = SliceOn<Self::Device>>,
    {
        softmax(PhantomData::<softmax::Scheme>, &self.softmax, att, queue);
    }

    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, queue: &QueueOf<Self::Device>)
    where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
    {
        swiglu(PhantomData::<swiglu::Scheme>, &self.swiglu, gate, up, queue);
    }
}
