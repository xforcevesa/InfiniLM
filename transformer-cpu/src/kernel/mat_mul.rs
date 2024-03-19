use gemm::{f16, gemm};
use std::ops::{Deref, DerefMut};
use tensor::{expand_indices, idx_strides, DataType, Tensor};

/// c = a x b
///
/// - c: [N0, N1, ... , N_, m, n]
/// - a: [N0, N1, ... , N_, m, k]
/// - b: [N0, N1, ... , N_, k, n]
pub fn mat_mul<T, U, V>(c: &mut Tensor<T>, beta: f32, a: &Tensor<U>, b: &Tensor<V>, alpha: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    let dt = c.data_type();
    assert_eq!(a.data_type(), dt);
    assert_eq!(b.data_type(), dt);

    #[cfg(detected_mkl)]
    {
        if dt == DataType::F32 {
            mkl::gemm(c, beta, a, b, alpha);
            return;
        }
    }

    let rank = c.shape().len();
    assert_eq!(a.shape().len(), rank);
    assert_eq!(b.shape().len(), rank);

    let (batch, mn) = c.shape().split_at(rank - 2);
    let (a_batch, mk) = a.shape().split_at(rank - 2);
    let (b_batch, kn) = b.shape().split_at(rank - 2);
    assert_eq!(a_batch, batch);
    assert_eq!(b_batch, batch);

    let m = mn[0];
    let n = mn[1];
    let k = mk[1];
    assert_eq!(mk, &[m, k]);
    assert_eq!(kn, &[k, n]);

    let dst_strides = &c.strides()[rank - 2..];
    let dst_cs = dst_strides[1] as isize;
    let dst_rs = dst_strides[0] as isize;

    let lhs_strides = &a.strides()[rank - 2..];
    let lhs_cs = lhs_strides[1] as isize;
    let lhs_rs = lhs_strides[0] as isize;

    let rhs_strides = &b.strides()[rank - 2..];
    let rhs_cs = rhs_strides[1] as isize;
    let rhs_rs = rhs_strides[0] as isize;

    let (batch, idx_strides) = idx_strides(batch);
    for i in 0..batch {
        let indices = expand_indices(i, &idx_strides, &[0, 0, 1]);
        let indices = indices.as_view();
        let dst = c.locate_mut(&indices).unwrap();
        let lhs = a.locate(&indices).unwrap();
        let rhs = b.locate(&indices).unwrap();
        match dt {
            DataType::F32 => unsafe {
                gemm(
                    m as _,
                    n as _,
                    k as _,
                    dst.cast::<f32>(),
                    dst_cs,
                    dst_rs,
                    beta != 0.,
                    lhs.cast::<f32>(),
                    lhs_cs,
                    lhs_rs,
                    rhs.cast::<f32>(),
                    rhs_cs,
                    rhs_rs,
                    beta,
                    alpha,
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(0),
                )
            },
            DataType::F16 => unsafe {
                gemm(
                    m as _,
                    n as _,
                    k as _,
                    dst.cast::<f16>(),
                    dst_cs,
                    dst_rs,
                    beta != 0.,
                    lhs.cast::<f16>(),
                    lhs_cs,
                    lhs_rs,
                    rhs.cast::<f16>(),
                    rhs_cs,
                    rhs_rs,
                    f16::from_f32(beta),
                    f16::from_f32(alpha),
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(0),
                )
            },
            _ => unreachable!(),
        }
    }
}

#[cfg(detected_mkl)]
mod mkl {
    use gemm::f16;
    use std::{
        mem::swap,
        ops::{Deref, DerefMut},
    };
    use tensor::Tensor;
    const COL_MAJOR: i32 = 102;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(C)]
    #[allow(non_camel_case_types)]
    pub enum CBLAS_TRANSPOSE {
        None = 111,
        Ordinary = 112,
    }

    extern "C" {
        fn cblas_hgemm(
            layout: i32,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: i32,
            n: i32,
            k: i32,
            alpha: f16,
            a: *const f16,
            lda: i32,
            b: *const f16,
            ldb: i32,
            beta: f16,
            c: *mut f16,
            ldc: i32,
        );

        fn cblas_sgemm(
            layout: i32,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    pub fn gemm<T, U, V>(c: &mut Tensor<T>, beta: f32, a: &Tensor<U>, b: &Tensor<V>, alpha: f32)
    where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        V: Deref<Target = [u8]>,
    {
        let dt = c.data_type();
        let mut c = Matrix::from(&*c);
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

        let m = c.r;
        let n = c.c;
        let k = a.c;

        let (lda, at) = a.ld_op();
        let (ldb, bt) = b.ld_op();
        let ldc = c.cs;

        assert_eq!(c.batch, a.batch);
        assert_eq!(c.batch, b.batch);

        match dt {
            tensor::DataType::F16 => unsafe {
                let alpha = f16::from_f32(alpha);
                let beta = f16::from_f32(beta);
                for i in 0..batch {
                    let a = a.ptr.cast::<f16>().offset((i * a.stride) as isize);
                    let b = b.ptr.cast::<f16>().offset((i * b.stride) as isize);
                    let c = c.ptr.cast::<f16>().offset((i * c.stride) as isize);
                    cblas_hgemm(
                        COL_MAJOR, at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    );
                }
            },
            tensor::DataType::F32 => unsafe {
                for i in 0..batch {
                    let a = a.ptr.cast::<f32>().offset((i * a.stride) as isize);
                    let b = b.ptr.cast::<f32>().offset((i * b.stride) as isize);
                    let c = c.ptr.cast::<f32>().offset((i * c.stride) as isize);
                    cblas_sgemm(
                        COL_MAJOR, at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    );
                }
            },
            _ => unreachable!(),
        }
    }

    struct Matrix {
        batch: i32,
        stride: i32,
        r: i32,
        c: i32,
        rs: i32,
        cs: i32,
        ptr: *mut u8,
    }

    impl<T> From<&Tensor<T>> for Matrix
    where
        T: Deref<Target = [u8]>,
    {
        fn from(tensor: &Tensor<T>) -> Self {
            let strides = tensor.strides();
            let ptr = tensor.locate_start().cast_mut();
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
        fn match_batch(&self, batch: i32) -> bool {
            self.batch == batch || self.batch == 1
        }

        #[inline]
        fn transpose(&mut self) {
            swap(&mut self.r, &mut self.c);
            swap(&mut self.rs, &mut self.cs);
        }

        #[inline]
        fn ld_op(&self) -> (i32, CBLAS_TRANSPOSE) {
            if self.rs == 1 {
                (self.cs, CBLAS_TRANSPOSE::None)
            } else if self.cs == 1 {
                (self.rs, CBLAS_TRANSPOSE::Ordinary)
            } else {
                panic!("Matrix is not contiguous");
            }
        }
    }
}
