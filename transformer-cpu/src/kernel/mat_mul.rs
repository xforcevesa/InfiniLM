use gemm::{f16, gemm};
use std::{
    ffi::{c_longlong, c_void},
    mem::swap,
    ops::{Deref, DerefMut},
};
use tensor::{DataType, Tensor};
use transformer::{BetweenF32, Matrix};

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

    #[inline]
    fn base(tensor: &impl Deref<Target = [u8]>) -> *mut c_void {
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

    assert_eq!(c.batch, a.batch);
    assert_eq!(c.batch, b.batch);
    match dt {
        DataType::F16 => gemm_as_blas::<f16>(c, beta, alpha, a, b),
        #[cfg(not(detected_mkl))]
        DataType::F32 => gemm_as_blas::<f32>(c, beta, alpha, a, b),
        #[cfg(detected_mkl)]
        DataType::F32 => mkl::gemm(dt, c, beta, alpha, a, b),
        _ => unreachable!(),
    }
}

fn gemm_as_blas<T: 'static + BetweenF32>(c: Matrix, beta: f32, alpha: f32, a: Matrix, b: Matrix) {
    let batch = c.batch;
    assert_eq!(a.batch, batch);
    assert_eq!(b.batch, batch);

    let m = c.r;
    let n = c.c;
    let k = a.c;
    assert_eq!(a.r, m);
    assert_eq!(b.r, k);
    assert_eq!(b.c, n);

    let c_ = c.base.cast::<T>();
    let a_ = a.base.cast::<T>();
    let b_ = b.base.cast::<T>();
    for i in 0..batch as c_longlong {
        unsafe {
            let c_ = c_.offset((i * c.stride) as isize);
            let a_ = a_.offset((i * a.stride) as isize);
            let b_ = b_.offset((i * b.stride) as isize);
            gemm(
                m as _,
                n as _,
                k as _,
                c_,
                c.cs as _,
                c.rs as _,
                beta != 0.,
                a_,
                a.cs as _,
                a.rs as _,
                b_,
                b.cs as _,
                b.rs as _,
                T::cast(beta),
                T::cast(alpha),
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            )
        }
    }
}

#[cfg(detected_mkl)]
mod mkl {
    use gemm::f16;
    use tensor::DataType;
    use transformer::Matrix;
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

    pub fn gemm(dt: DataType, c: Matrix, beta: f32, alpha: f32, a: Matrix, b: Matrix) {
        let batch = c.batch;
        assert_eq!(a.batch, batch);
        assert_eq!(b.batch, batch);

        let m = c.r;
        let n = c.c;
        let k = a.c;
        assert_eq!(a.r, m);
        assert_eq!(b.r, k);
        assert_eq!(b.c, n);

        #[inline]
        fn trans(m: &Matrix) -> CBLAS_TRANSPOSE {
            if m.rs == 1 {
                CBLAS_TRANSPOSE::None
            } else if m.cs == 1 {
                CBLAS_TRANSPOSE::Ordinary
            } else {
                panic!("Matrix is not contiguous");
            }
        }

        match dt {
            DataType::F16 => unsafe {
                let alpha = f16::from_f32(alpha);
                let beta = f16::from_f32(beta);
                for i in 0..batch {
                    let a_ = a.base.cast::<f16>().offset(i as isize * a.stride as isize);
                    let b_ = b.base.cast::<f16>().offset(i as isize * b.stride as isize);
                    let c_ = c.base.cast::<f16>().offset(i as isize * c.stride as isize);
                    cblas_hgemm(
                        COL_MAJOR,
                        trans(&a),
                        trans(&b),
                        m,
                        n,
                        k,
                        alpha,
                        a_,
                        a.ld(),
                        b_,
                        b.ld(),
                        beta,
                        c_,
                        c.ld(),
                    );
                }
            },
            DataType::F32 => unsafe {
                for i in 0..batch {
                    let a_ = a.base.cast::<f32>().offset(i as isize * a.stride as isize);
                    let b_ = b.base.cast::<f32>().offset(i as isize * b.stride as isize);
                    let c_ = c.base.cast::<f32>().offset(i as isize * c.stride as isize);
                    cblas_sgemm(
                        COL_MAJOR,
                        trans(&a),
                        trans(&b),
                        m,
                        n,
                        k,
                        alpha,
                        a_,
                        a.ld(),
                        b_,
                        b.ld(),
                        beta,
                        c_,
                        c.ld(),
                    );
                }
            },
            _ => unreachable!(),
        }
    }
}
