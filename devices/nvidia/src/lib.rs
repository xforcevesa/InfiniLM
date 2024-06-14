#![cfg(detected_cuda)]

pub extern crate cuda;

mod gather;
mod sample;

use common::utok;
use common_devices::{layout, rms_norm};
use cuda::{CudaDataType, DevByte, Device, Stream};
use operators::{
    fuesd_softmax::{self, nvidia_gpu as softmax_nv},
    mat_mul::{self, nvidia_gpu as mat_mul_nv},
    reform::{self, nvidia_gpu as reform_nv},
    rms_norm::nvidia_gpu as rms_norm_nv,
    rope::{self, nvidia_gpu as rope_nv},
    swiglu::{self, nvidia_gpu as swiglu_nv},
    Operator, Scheme, F16,
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
        Self {
            mat_mul: mat_mul_nv::Operator::new(&F16).unwrap(),
            rms_norm: rms_norm_nv::Operator::new(&rms_norm_nv::Config::config_for(
                &devices[0],
                F16,
                rms_norm_max_size,
            ))
            .unwrap(),
            rope: rope_nv::Operator::new(&rope_nv::Config::config_for(&devices[0], F16)).unwrap(),
            reform: reform_nv::Operator::new(&reform_nv::Config::config_for(&devices[0])).unwrap(),
            softmax: softmax_nv::Operator::new(&softmax_nv::Config::config_for(
                &devices[0],
                F16,
                softmax_max_size,
            ))
            .unwrap(),
            swiglu: swiglu_nv::Operator::new(&swiglu_nv::Config::config_for(&devices[0], F16))
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
        mat_mul_nv::Scheme::new(
            &self.mat_mul,
            mat_mul::LayoutAttrs {
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
            stream,
        );
    }

    #[inline]
    pub fn rotary_embedding<T, U>(
        &self,
        t: &mut Tensor<T>,
        pos: &Tensor<U>,
        theta: f32,
        stream: &Stream,
    ) where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        rope_nv::Scheme::new(
            &self.rope,
            rope::LayoutAttrs {
                t: layout(t),
                pos: layout(pos),
            },
        )
        .unwrap()
        .launch(
            &(
                t.physical_mut().as_mut_ptr(),
                pos.physical().as_ptr(),
                theta,
            ),
            stream,
        );
    }

    #[inline]
    pub fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        reform_nv::Scheme::new(
            &self.reform,
            reform::LayoutAttrs {
                dst: layout(dst),
                src: layout(src),
            },
        )
        .unwrap()
        .launch(
            &(dst.physical_mut().as_mut_ptr(), src.physical().as_ptr()),
            stream,
        );
    }

    #[inline]
    pub fn softmax<T>(&self, att: &mut Tensor<T>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
    {
        softmax_nv::Scheme::new(
            &self.softmax,
            fuesd_softmax::LayoutAttrs { att: layout(att) },
        )
        .unwrap()
        .launch(&att.physical_mut().as_mut_ptr(), stream);
    }

    #[inline]
    pub fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        swiglu_nv::Scheme::new(
            &self.swiglu,
            swiglu::LayoutAttrs {
                gate: layout(gate),
                up: layout(up),
            },
        )
        .unwrap()
        .launch(
            &(gate.physical_mut().as_mut_ptr(), up.physical().as_ptr()),
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
