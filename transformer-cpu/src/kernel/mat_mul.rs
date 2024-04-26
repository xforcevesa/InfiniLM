use common::{f16, BetweenF32};
use gemm::gemm;
use std::{
    ffi::{c_longlong, c_void},
    mem::swap,
    ops::{Deref, DerefMut},
};
use tensor::{DataType, Tensor};
use transformer::Matrix;

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
    // const LAYER: usize = 40;
    // static mut I: usize = 0;
    // unsafe {
    //     if I == 0 {
    //         println!();
    //         // #[cfg(detected_mkl)]
    //         // {
    //         //     println!("MKL threads: {}", mkl::mkl_get_max_threads());
    //         //     println!("MKL dynamic: {}", mkl::mkl_get_dynamic());
    //         // }
    //     }
    // }
    // let time = std::time::Instant::now();
    match dt {
        // DataType::F16 => mkl::gemm(dt, c, beta, alpha, a, b),
        DataType::F16 => gemm_as_blas::<f16>(c, beta, alpha, a, b),
        #[cfg(not(detected_mkl))]
        DataType::F32 => gemm_as_blas::<f32>(c, beta, alpha, a, b),
        #[cfg(detected_mkl)]
        DataType::F32 => mkl::gemm(dt, c, beta, alpha, a, b),
        _ => unreachable!(),
    }
    // unsafe {
    //     if I % 6 == 0 {
    //         println!();
    //     }
    //     println!("{}:{} {}", I / 6, I % 6, time.elapsed().as_micros());
    //     if I == LAYER * 6 {
    //         I = 0;
    //     } else {
    //         I += 1;
    //     }
    // }
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
    use std::ffi::c_int;
    use tensor::DataType;
    use transformer::Matrix;
    const COL_MAJOR: c_int = 102;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(C)]
    #[allow(non_camel_case_types)]
    pub enum CBLAS_TRANSPOSE {
        None = 111,
        Ordinary = 112,
    }

    #[allow(unused)]
    extern "C" {
        pub fn mkl_get_max_threads() -> c_int;
        pub fn mkl_get_dynamic() -> c_int;
        pub fn mkl_set_num_threads(nt: c_int);
        pub fn mkl_set_num_threads_local(nt: c_int);

        fn cblas_hgemm(
            layout: c_int,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f16,
            a: *const f16,
            lda: c_int,
            b: *const f16,
            ldb: c_int,
            beta: f16,
            c: *mut f16,
            ldc: c_int,
        );

        fn cblas_sgemm(
            layout: c_int,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
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
                for i in 0..batch as isize {
                    let a_ = a.base.cast::<f16>().offset(i * a.stride as isize);
                    let b_ = b.base.cast::<f16>().offset(i * b.stride as isize);
                    let c_ = c.base.cast::<f16>().offset(i * c.stride as isize);
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
                for i in 0..batch as isize {
                    let a_ = a.base.cast::<f32>().offset(i * a.stride as isize);
                    let b_ = b.base.cast::<f32>().offset(i * b.stride as isize);
                    let c_ = c.base.cast::<f32>().offset(i * c.stride as isize);
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
