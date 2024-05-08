#![cfg(detected_cuda)]

#[macro_use]
extern crate log;
pub extern crate cuda;

mod fused_softmax;
mod gather;
mod mat_mul;
mod reform;
mod rms_norm;
mod rotary_embedding;
mod sample;
mod swiglu;

use common::utok;
use cublas::{Cublas, CublasSpore};
use cuda::{
    memcpy_d2h, ContextGuard, ContextResource, ContextSpore, CudaDataType, DevByte, ModuleSpore,
    Ptx, Stream,
};
use fused_softmax::FusedSoftmax;
use reform::Reform;
use rms_norm::RmsNormalization;
use rotary_embedding::Rope;
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};
use swiglu::Swiglu;

pub use kernel_lib::Kernels;
pub use sample::{sample_cpu, sample_nv};
pub use tensor::{reslice, reslice_mut, slice, split, udim, DataType, LocalSplitable, Tensor};

pub struct NvidiaKernelsPtx {
    rms_norm: Arc<RmsNormalization>,
    rotary_embedding: Arc<Rope>,
    reform: Arc<Reform>,
    softmax: Arc<FusedSoftmax>,
    swiglu: Arc<Swiglu>,
}

impl NvidiaKernelsPtx {
    pub fn new(rms_norm_max_size: usize, softmax_max_size: usize, block_size: usize) -> Self {
        Self {
            rms_norm: Arc::new(RmsNormalization::new(
                CudaDataType::f16,
                rms_norm_max_size,
                block_size,
            )),
            rotary_embedding: Arc::new(Rope::new(block_size)),
            reform: Arc::new(Reform::new(block_size, 32)),
            softmax: Arc::new(FusedSoftmax::new(
                CudaDataType::f16,
                softmax_max_size,
                block_size,
            )),
            swiglu: Arc::new(Swiglu::new(CudaDataType::f16, block_size)),
        }
    }
}

trait PtxWapper: Sized {
    fn ptx(&self) -> &Ptx;
    #[inline]
    fn load(self: Arc<Self>, ctx: &ContextGuard) -> ModuleWapper<Self> {
        ModuleWapper {
            module: ctx.load(self.ptx()).sporulate(),
            kernel: self,
        }
    }
}

struct ModuleWapper<T> {
    module: ModuleSpore,
    kernel: Arc<T>,
}

pub struct NvidiaKernels {
    cublas: CublasSpore,
    rms_norm: ModuleWapper<RmsNormalization>,
    rotary_embedding: ModuleWapper<Rope>,
    reform: ModuleWapper<Reform>,
    softmax: ModuleWapper<FusedSoftmax>,
    swiglu: ModuleWapper<Swiglu>,
}

impl NvidiaKernelsPtx {
    pub fn load(&self, stream: &Stream) -> NvidiaKernels {
        let ctx = stream.ctx();
        let cublas = Cublas::new(ctx);
        cublas.set_stream(stream);
        NvidiaKernels {
            cublas: cublas.sporulate(),
            rms_norm: self.rms_norm.clone().load(ctx),
            rotary_embedding: self.rotary_embedding.clone().load(ctx),
            reform: self.reform.clone().load(ctx),
            softmax: self.softmax.clone().load(ctx),
            swiglu: self.swiglu.clone().load(ctx),
        }
    }
}

impl NvidiaKernels {
    pub fn kill(&mut self, ctx: &ContextGuard) {
        unsafe {
            self.cublas.kill(ctx);
            self.rms_norm.module.kill(ctx);
            self.rotary_embedding.module.kill(ctx);
            self.reform.module.kill(ctx);
            self.softmax.module.kill(ctx);
            self.swiglu.module.kill(ctx);
        }
    }
}

pub struct KernelRuntime<'a> {
    pub kernels: &'a NvidiaKernels,
    pub stream: &'a Stream<'a>,
}

impl NvidiaKernels {
    #[inline]
    pub fn on<'a>(&'a self, stream: &'a Stream) -> KernelRuntime<'a> {
        KernelRuntime {
            kernels: self,
            stream,
        }
    }
}

impl Kernels for KernelRuntime<'_> {
    type Storage = [DevByte];

    #[inline]
    fn gather<T, U, I>(&self, x: &mut Tensor<T>, table: &Tensor<U>, tokens: I)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, self.stream);
    }

    #[inline]
    fn rms_norm<T, U, V>(&self, y: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>, epsilon: f32)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
        V: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.rms_norm;
        kernel.launch(module, y, x, w, epsilon, self.stream);
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
        let cublas = unsafe { self.kernels.cublas.sprout(self.stream.ctx()) };
        mat_mul::mat_mul(&cublas, c, beta, a, b, alpha)
    }

    #[inline]
    fn rotary_embedding<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.rotary_embedding;
        kernel.launch(module, t, pos, theta, self.stream);
    }

    #[inline]
    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.reform;
        kernel.launch(module, dst, src, self.stream);
    }

    #[inline]
    fn softmax<T>(&self, att: &mut Tensor<T>)
    where
        T: DerefMut<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.softmax;
        kernel.launch(module, att, self.stream);
    }

    #[inline]
    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.swiglu;
        kernel.launch(module, gate, up, self.stream);
    }
}

#[inline]
pub fn cast_dt(dt: DataType) -> CudaDataType {
    match dt {
        DataType::I8 => CudaDataType::i8,
        DataType::I16 => CudaDataType::i16,
        DataType::I32 => CudaDataType::i32,
        DataType::I64 => CudaDataType::i64,
        DataType::U8 => CudaDataType::u8,
        DataType::U16 => CudaDataType::u16,
        DataType::U32 => CudaDataType::u32,
        DataType::U64 => CudaDataType::u64,
        DataType::F16 => CudaDataType::f16,
        DataType::BF16 => CudaDataType::bf16,
        DataType::F32 => CudaDataType::f32,
        DataType::F64 => CudaDataType::f64,
        _ => unreachable!(),
    }
}

#[allow(unused)]
pub fn map_tensor<T>(tensor: &Tensor<T>) -> Tensor<Vec<u8>>
where
    T: Deref<Target = [DevByte]>,
{
    unsafe {
        tensor.as_ref().map_physical(|dev| {
            let mut buf = vec![0; dev.len()];
            memcpy_d2h(&mut buf, dev);
            buf
        })
    }
}

pub fn synchronize() {
    cuda::init();
    for i in 0..cuda::Device::count() {
        cuda::Device::new(i as _)
            .retain_primary()
            .apply(|ctx| ctx.synchronize());
    }
}
