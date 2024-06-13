use crate::layout;
use operators::{
    common_cpu::ThisThread,
    mat_mul::{
        common_cpu::{Operator as MatMul, Scheme as MatMulScheme},
        LayoutAttrs,
    },
    Operator, Scheme, F16,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

/// c = a x b
pub fn mat_mul<T, U, V>(c: &mut Tensor<T>, beta: f32, a: &Tensor<U>, b: &Tensor<V>, alpha: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    MatMulScheme::new(
        &MatMul::new(&F16).unwrap(),
        LayoutAttrs {
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
        &ThisThread,
    );
}

#[cfg(detected_mkl)]
#[allow(unused)]
mod mkl {
    use common::f16;
    use kernel_lib::Matrix;
    use std::ffi::c_int;
    use tensor::DataType;
    const COL_MAJOR: c_int = 102;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(C)]
    #[allow(non_camel_case_types)]
    pub enum CBLAS_TRANSPOSE {
        None = 111,
        Ordinary = 112,
    }

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
