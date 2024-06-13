#![cfg(detected_cuda)]

pub extern crate cuda;

mod gather;
mod sample;

use common::utok;
use cuda::{memcpy_d2h, CudaDataType, DevByte, Device, Stream};
use operators::{
    fuesd_softmax::{self, nvidia_gpu as softmax_nv},
    mat_mul::{self, nvidia_gpu as mat_mul_nv},
    reform::{self, nvidia_gpu as reform_nv},
    rms_norm::{self, nvidia_gpu as rms_norm_nv},
    rope::{self, nvidia_gpu as rope_nv},
    swiglu::{self, nvidia_gpu as swiglu_nv},
    Operator, Scheme, TensorLayout, F16, U32,
};
use std::ops::{Deref, DerefMut};

pub use kernel_lib::Kernels;
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
        rms_norm_nv::Scheme::new(
            &self.kernels.rms_norm,
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
            self.stream,
        );
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
        mat_mul_nv::Scheme::new(
            &self.kernels.mat_mul,
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
            self.stream,
        );
    }

    #[inline]
    fn rotary_embedding<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        rope_nv::Scheme::new(
            &self.kernels.rope,
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
            self.stream,
        );
    }

    #[inline]
    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        reform_nv::Scheme::new(
            &self.kernels.reform,
            reform::LayoutAttrs {
                dst: layout(dst),
                src: layout(src),
            },
        )
        .unwrap()
        .launch(
            &(dst.physical_mut().as_mut_ptr(), src.physical().as_ptr()),
            self.stream,
        );
    }

    #[inline]
    fn softmax<T>(&self, att: &mut Tensor<T>)
    where
        T: DerefMut<Target = Self::Storage>,
    {
        softmax_nv::Scheme::new(
            &self.kernels.softmax,
            fuesd_softmax::LayoutAttrs { att: layout(att) },
        )
        .unwrap()
        .launch(&att.physical_mut().as_mut_ptr(), self.stream);
    }

    #[inline]
    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        swiglu_nv::Scheme::new(
            &self.kernels.swiglu,
            swiglu::LayoutAttrs {
                gate: layout(gate),
                up: layout(up),
            },
        )
        .unwrap()
        .launch(
            &(gate.physical_mut().as_mut_ptr(), up.physical().as_ptr()),
            self.stream,
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

fn layout<T>(t: &Tensor<T>) -> TensorLayout {
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
