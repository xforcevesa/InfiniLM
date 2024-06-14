#![cfg(detected_cuda)]

pub extern crate cuda;

mod gather;
mod sample;

use common::utok;
use common_devices::{mat_mul, reform, rms_norm, rope, softmax, swiglu, SliceOn};
use cuda::{CudaDataType, Device};
use operators::{
    fuesd_softmax::nvidia_gpu as softmax, mat_mul::nvidia_gpu as mat_mul,
    reform::nvidia_gpu as reform, rms_norm::nvidia_gpu as rms_norm, rope::nvidia_gpu as rope,
    swiglu::nvidia_gpu as swiglu, Operator, QueueOf, F16,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub use common_devices::Kernels;
pub use operators::nvidia_gpu::Device as Gpu;
pub use sample::{sample_cpu, sample_nv};
pub use tensor::{reslice, reslice_mut, slice, split, udim, DataType, LocalSplitable, Tensor};

pub struct NvidiaKernels {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    reform: reform::Operator,
    softmax: softmax::Operator,
    swiglu: swiglu::Operator,
}

impl NvidiaKernels {
    pub fn new(devices: &[Device], rms_norm_max_size: usize, softmax_max_size: usize) -> Self {
        let max_num_threads_block = devices.iter().map(|d| d.max_block_dims().0).min().unwrap();
        let compute_capability = devices
            .iter()
            .map(Device::compute_capability)
            .min()
            .unwrap();
        Self {
            mat_mul: mat_mul::Operator::new(&F16).unwrap(),
            rms_norm: rms_norm::Operator::new(&rms_norm::Config {
                data_layout: F16,
                num_items_reduce: rms_norm_max_size,
                num_threads_warp: 32,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            rope: rope::Operator::new(&rope::Config {
                data_layout: F16,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            reform: reform::Operator::new(&reform::Config {
                num_threads_warp: 32,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            softmax: softmax::Operator::new(&softmax::Config {
                data_layout: F16,
                max_seq_len: softmax_max_size,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            swiglu: swiglu::Operator::new(&swiglu::Config {
                data_layout: F16,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
        }
    }
}

impl Kernels for NvidiaKernels {
    type Device = Gpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        queue: &QueueOf<Self::Device>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, queue);
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

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, queue: &QueueOf<Self::Device>)
    where
        T: DerefMut<Target = SliceOn<Self::Device>>,
        U: Deref<Target = SliceOn<Self::Device>>,
    {
        reform(PhantomData::<reform::Scheme>, &self.reform, dst, src, queue);
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

pub fn synchronize() {
    cuda::init();
    for i in 0..cuda::Device::count() {
        cuda::Device::new(i as _)
            .retain_primary()
            .apply(|ctx| ctx.synchronize());
    }
}
