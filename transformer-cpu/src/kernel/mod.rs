mod fused_softmax;
mod gather;
mod mat_mul;
mod rms_norm;
mod rotary_embedding;
mod swiglu;

macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

pub(super) use slice;

use common::utok;
use std::ops::{Deref, DerefMut};
use tensor::Tensor;
use transformer::{Kernels, Llama2};

pub struct CpuKernels {
    epsilon: f32,
    theta: f32,
}

impl CpuKernels {
    #[inline]
    pub fn new(model: &dyn Llama2) -> Self {
        Self {
            epsilon: model.rms_norm_eps(),
            theta: model.rope_theta(),
        }
    }
}

impl Kernels for CpuKernels {
    type Storage = [u8];

    #[inline]
    fn gather<T, U, I>(&self, x: &mut Tensor<T>, table: &Tensor<U>, tokens: I)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens);
    }

    #[inline]
    fn rms_norm<T, U, V>(&self, y: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
        V: Deref<Target = Self::Storage>,
    {
        rms_norm::rms_norm(y, x, w, self.epsilon);
    }

    #[inline]
    fn mat_mul<T, U, V>(
        &self,
        c: &mut Tensor<T>,
        beta: f32,
        a: &Tensor<U>,
        b: &Tensor<V>,
        alpha: f32,
    ) where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
        V: Deref<Target = Self::Storage>,
    {
        mat_mul::mat_mul(c, beta, a, b, alpha);
    }

    #[inline]
    fn rotary_embedding<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        rotary_embedding::rotary_embedding(t, pos, self.theta);
    }

    #[inline]
    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        src.reform_to(dst);
    }

    #[inline]
    fn softmax<T>(&self, att: &mut Tensor<T>)
    where
        T: DerefMut<Target = Self::Storage>,
    {
        fused_softmax::softmax(att);
    }

    #[inline]
    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        swiglu::swiglu(gate, up);
    }
}
