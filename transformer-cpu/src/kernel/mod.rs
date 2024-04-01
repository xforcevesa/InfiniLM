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
    fn gather<T, U, I>(&self, x: &mut tensor::Tensor<T>, table: &tensor::Tensor<U>, tokens: I)
    where
        T: std::ops::DerefMut<Target = Self::Storage>,
        U: std::ops::Deref<Target = [u8]>,
        I: IntoIterator<Item = common::utok>,
    {
        gather::gather(x, table, tokens);
    }

    #[inline]
    fn rms_norm<T, U, V>(
        &self,
        y: &mut tensor::Tensor<T>,
        x: &tensor::Tensor<U>,
        w: &tensor::Tensor<V>,
    ) where
        T: std::ops::DerefMut<Target = Self::Storage>,
        U: std::ops::Deref<Target = Self::Storage>,
        V: std::ops::Deref<Target = Self::Storage>,
    {
        rms_norm::rms_norm(y, x, w, self.epsilon);
    }

    #[inline]
    fn mat_mul<T, U, V>(
        &self,
        c: &mut tensor::Tensor<T>,
        beta: f32,
        a: &tensor::Tensor<U>,
        b: &tensor::Tensor<V>,
        alpha: f32,
    ) where
        T: std::ops::DerefMut<Target = Self::Storage>,
        U: std::ops::Deref<Target = Self::Storage>,
        V: std::ops::Deref<Target = Self::Storage>,
    {
        mat_mul::mat_mul(c, beta, a, b, alpha);
    }

    #[inline]
    fn rotary_embedding<T, U>(&self, t: &mut tensor::Tensor<T>, pos: &tensor::Tensor<U>)
    where
        T: std::ops::DerefMut<Target = Self::Storage>,
        U: std::ops::Deref<Target = Self::Storage>,
    {
        rotary_embedding::rotary_embedding(t, pos, self.theta);
    }

    #[inline]
    fn reform<T, U>(&self, dst: &mut tensor::Tensor<T>, src: &tensor::Tensor<U>)
    where
        T: std::ops::DerefMut<Target = Self::Storage>,
        U: std::ops::Deref<Target = Self::Storage>,
    {
        src.reform_to(dst);
    }

    #[inline]
    fn softmax<T>(&self, att: &mut tensor::Tensor<T>)
    where
        T: std::ops::DerefMut<Target = Self::Storage>,
    {
        fused_softmax::softmax(att);
    }

    #[inline]
    fn swiglu<T, U>(&self, gate: &mut tensor::Tensor<T>, up: &tensor::Tensor<U>)
    where
        T: std::ops::DerefMut<Target = Self::Storage>,
        U: std::ops::Deref<Target = Self::Storage>,
    {
        swiglu::swiglu(gate, up);
    }
}
