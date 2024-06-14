#[macro_export]
macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

mod gather;

use common::utok;
use common_devices::{mat_mul, rms_norm, rope, softmax, swiglu};
use operators::{
    fuesd_softmax::common_cpu as softmax, mat_mul::common_cpu as mat_mul,
    rms_norm::common_cpu as rms_norm, rope::common_cpu as rope, swiglu::common_cpu as swiglu,
    Operator, F16,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};
use tensor::Tensor;

pub struct CpuKernels {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    softmax: softmax::Operator,
    swiglu: swiglu::Operator,
}
pub use operators::common_cpu::ThisThread;

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

impl CpuKernels {
    #[inline]
    pub fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        _queue: &ThisThread,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens);
    }

    #[inline]
    pub fn rms_norm<T, U, V>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<U>,
        w: &Tensor<V>,
        epsilon: f32,
        queue: &ThisThread,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        V: Deref<Target = [u8]>,
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

    #[inline]
    pub fn rope<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32, queue: &ThisThread)
    where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
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

    #[inline]
    pub fn mat_mul<T, U, V>(
        &self,
        c: &mut Tensor<T>,
        beta: f32,
        a: &Tensor<U>,
        b: &Tensor<V>,
        alpha: f32,
        queue: &ThisThread,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        V: Deref<Target = [u8]>,
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

    #[inline]
    pub fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, _queue: &ThisThread)
    where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
    {
        src.reform_to(dst);
    }

    #[inline]
    pub fn softmax<T>(&self, att: &mut Tensor<T>, queue: &ThisThread)
    where
        T: DerefMut<Target = [u8]>,
    {
        softmax(PhantomData::<softmax::Scheme>, &self.softmax, att, queue);
    }

    #[inline]
    pub fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, queue: &ThisThread)
    where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
    {
        swiglu(PhantomData::<swiglu::Scheme>, &self.swiglu, gate, up, queue);
    }
}
