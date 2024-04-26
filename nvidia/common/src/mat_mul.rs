use common::f16;
use cublas::{bindings::cublasOperation_t, cublas, Cublas};
use cuda::{AsRaw, DevByte};
use std::{
    mem::swap,
    ops::{Deref, DerefMut},
    os::raw::c_void,
};
use tensor::{DataType, Tensor};
use transformer::Matrix;

pub fn mat_mul<T, U, V>(
    handle: &Cublas,
    c: &mut Tensor<T>,
    beta: f32,
    a: &Tensor<U>,
    b: &Tensor<V>,
    alpha: f32,
) where
    T: DerefMut<Target = [DevByte]>,
    U: Deref<Target = [DevByte]>,
    V: Deref<Target = [DevByte]>,
{
    let dt = c.data_type();
    assert_eq!(dt, DataType::F16);
    assert_eq!(a.data_type(), dt);
    assert_eq!(b.data_type(), dt);

    #[inline]
    fn base(tensor: &impl Deref<Target = [DevByte]>) -> *mut c_void {
        tensor.as_ptr() as _
    }

    let mut c = Matrix::new(c, base);
    let mut a = Matrix::new(a, base);
    let mut b = Matrix::new(b, base);
    assert_eq!(c.r, a.r); // m
    assert_eq!(c.c, b.c); // n
    assert_eq!(a.c, b.r); // k

    let batch = c.batch;
    if !a.match_batch(batch) || !b.match_batch(batch) {
        panic!("Invalid batch size");
    }

    if c.rs == 1 {
        // Nothing to do
    } else if c.cs == 1 {
        c.transpose();
        a.transpose();
        b.transpose();
        swap(&mut a, &mut b);
    } else {
        panic!("Matrix is not contiguous");
    };

    let alpha = f16::from_f32(alpha);
    let beta = f16::from_f32(beta);

    let m = c.r;
    let n = c.c;
    let k = a.c;

    #[inline]
    fn trans(m: &Matrix) -> cublasOperation_t {
        if m.rs == 1 {
            cublasOperation_t::CUBLAS_OP_N
        } else if m.cs == 1 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            panic!("Matrix is not contiguous");
        }
    }

    cublas!(cublasGemmStridedBatchedEx(
        handle.as_raw(),
        trans(&a),
        trans(&b),
        m,
        n,
        k,
        ((&alpha) as *const f16).cast(),
        a.base as _,
        cudaDataType_t::CUDA_R_16F,
        a.ld(),
        a.stride,
        b.base as _,
        cudaDataType_t::CUDA_R_16F,
        b.ld(),
        b.stride,
        ((&beta) as *const f16).cast(),
        c.base as _,
        cudaDataType_t::CUDA_R_16F,
        c.ld(),
        c.stride,
        batch,
        cublasComputeType_t::CUBLAS_COMPUTE_16F,
        cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
    ));
}
