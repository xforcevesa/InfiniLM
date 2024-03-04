use gemm::{f16, gemm};
use std::ops::{Deref, DerefMut};
use tensor::{expand_indices, idx_strides, DataType, Tensor};

/// c = a x b
///
/// - c: [N0, N1, ... , N_, m, n]
/// - a: [N0, N1, ... , N_, m, k]
/// - b: [N0, N1, ... , N_, k, n]
pub fn mat_mul<T, U, V>(mut c: Tensor<T>, beta: f32, a: &Tensor<U>, b: &Tensor<V>, alpha: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    let dt = c.data_type();
    assert_eq!(a.data_type(), dt);
    assert_eq!(b.data_type(), dt);

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
