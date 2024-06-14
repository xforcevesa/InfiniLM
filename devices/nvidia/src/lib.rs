#![cfg(detected_cuda)]

pub extern crate cuda;

mod gather;
mod sample;

use common::utok;
use common_devices::{mat_mul, reform, rms_norm, rope, softmax, swiglu};
use cuda::{CudaDataType, DevByte, Device, Stream};
use operators::{
    fuesd_softmax::nvidia_gpu as softmax_nv, mat_mul::nvidia_gpu as mat_mul_nv,
    reform::nvidia_gpu as reform_nv, rms_norm::nvidia_gpu as rms_norm_nv,
    rope::nvidia_gpu as rope_nv, swiglu::nvidia_gpu as swiglu_nv, Operator, F16,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub use sample::{sample_cpu, sample_nv};
pub use tensor::{reslice, reslice_mut, slice, split, udim, DataType, LocalSplitable, Tensor};

pub struct NvidiaKernels {
    mat_mul: mat_mul_nv::Operator,
    rms_norm: rms_norm_nv::Operator,
    rope: rope_nv::Operator,
    reform: reform_nv::Operator,
    softmax: softmax_nv::Operator,
    swiglu: swiglu_nv::Operator,
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
            mat_mul: mat_mul_nv::Operator::new(&F16).unwrap(),
            rms_norm: rms_norm_nv::Operator::new(&rms_norm_nv::Config {
                data_layout: F16,
                num_items_reduce: rms_norm_max_size,
                num_threads_warp: 32,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            rope: rope_nv::Operator::new(&rope_nv::Config {
                data_layout: F16,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            reform: reform_nv::Operator::new(&reform_nv::Config {
                num_threads_warp: 32,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            softmax: softmax_nv::Operator::new(&softmax_nv::Config {
                data_layout: F16,
                max_seq_len: softmax_max_size,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
            swiglu: swiglu_nv::Operator::new(&swiglu_nv::Config {
                data_layout: F16,
                max_num_threads_block,
                compute_capability,
            })
            .unwrap(),
        }
    }
}

impl NvidiaKernels {
    #[inline]
    pub fn gather<T, U, I>(&self, x: &mut Tensor<T>, table: &Tensor<U>, tokens: I, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, stream);
    }

    #[inline]
    pub fn rms_norm<T, U, V>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<U>,
        w: &Tensor<V>,
        epsilon: f32,
        stream: &Stream,
    ) where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
        V: Deref<Target = [DevByte]>,
    {
        rms_norm(
            PhantomData::<rms_norm_nv::Scheme>,
            &self.rms_norm,
            y,
            x,
            w,
            epsilon,
            stream,
        );
    }

    #[inline]
    pub fn rope<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        rope(
            PhantomData::<rope_nv::Scheme>,
            &self.rope,
            t,
            pos,
            theta,
            stream,
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
        stream: &Stream,
    ) where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
        V: Deref<Target = [DevByte]>,
    {
        mat_mul(
            PhantomData::<mat_mul_nv::Scheme>,
            &self.mat_mul,
            c,
            beta,
            a,
            b,
            alpha,
            stream,
        );
    }

    #[inline]
    pub fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        reform(
            PhantomData::<reform_nv::Scheme>,
            &self.reform,
            dst,
            src,
            stream,
        );
    }

    #[inline]
    pub fn softmax<T>(&self, att: &mut Tensor<T>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
    {
        softmax(
            PhantomData::<softmax_nv::Scheme>,
            &self.softmax,
            att,
            stream,
        );
    }

    #[inline]
    pub fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        swiglu(
            PhantomData::<swiglu_nv::Scheme>,
            &self.swiglu,
            gate,
            up,
            stream,
        );
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
