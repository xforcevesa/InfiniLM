use cublas::{bindings::cublasOperation_t, cublas, Cublas};
use cuda::{AsRaw, DevMem};
use half::f16;
use std::{
    ffi::{c_int, c_longlong, c_void},
    mem::swap,
    ops::Deref,
};
use tensor::{DataType, Tensor};

pub fn mat_mul<'a, T>(
    handle: &Cublas,
    c: &Tensor<T>,
    beta: f32,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: f32,
) where
    T: Deref<Target = DevMem<'a>>,
{
    assert_eq!(c.data_type(), DataType::F16);
    assert_eq!(a.data_type(), DataType::F16);
    assert_eq!(b.data_type(), DataType::F16);

    let mut c = Matrix::from(c);
    let mut a = Matrix::from(a);
    let mut b = Matrix::from(b);
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

    let (lda, transa) = a.ld_op();
    let (ldb, transb) = b.ld_op();
    let ldc = c.cs;

    cublas!(cublasGemmStridedBatchedEx(
        handle.as_raw(),
        transa,
        transb,
        m,
        n,
        k,
        ((&alpha) as *const f16).cast(),
        a.ptr,
        cudaDataType_t::CUDA_R_16F,
        lda,
        a.stride,
        b.ptr,
        cudaDataType_t::CUDA_R_16F,
        ldb,
        b.stride,
        ((&beta) as *const f16).cast(),
        c.ptr,
        cudaDataType_t::CUDA_R_16F,
        ldc,
        c.stride,
        batch,
        cublasComputeType_t::CUBLAS_COMPUTE_16F,
        cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
    ));
}

struct Matrix {
    batch: c_int,
    stride: c_longlong,
    r: c_int,
    c: c_int,
    rs: c_int,
    cs: c_int,
    ptr: *mut c_void,
}

impl<'a, T> From<&Tensor<T>> for Matrix
where
    T: Deref<Target = DevMem<'a>>,
{
    fn from(tensor: &Tensor<T>) -> Self {
        let strides = tensor.strides();
        let ptr = (unsafe { tensor.physical().as_raw() } as isize + tensor.bytes_offset()) as _;
        match tensor.shape() {
            &[r, c] => Self {
                batch: 1,
                stride: 0,
                r: r as _,
                c: c as _,
                rs: strides[0] as _,
                cs: strides[1] as _,
                ptr,
            },
            &[batch, r, c] => Self {
                batch: batch as _,
                stride: if batch == 1 { 0 } else { strides[0] as _ },
                r: r as _,
                c: c as _,
                rs: strides[1] as _,
                cs: strides[2] as _,
                ptr,
            },
            s => panic!("Invalid matrix shape: {s:?}"),
        }
    }
}

impl Matrix {
    #[inline]
    fn match_batch(&self, batch: c_int) -> bool {
        self.batch == batch || self.batch == 1
    }

    #[inline]
    fn transpose(&mut self) {
        swap(&mut self.r, &mut self.c);
        swap(&mut self.rs, &mut self.cs);
    }

    #[inline]
    fn ld_op(&self) -> (c_int, cublasOperation_t) {
        if self.rs == 1 {
            (self.cs, cublasOperation_t::CUBLAS_OP_N)
        } else if self.cs == 1 {
            (self.rs, cublasOperation_t::CUBLAS_OP_T)
        } else {
            panic!("Matrix is not contiguous");
        }
    }
}
