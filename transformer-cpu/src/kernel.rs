use common::utok;
use gemm::{f16, gemm};
use std::{
    iter::zip,
    ops::{Deref, DerefMut, Mul, MulAssign},
};
use tensor::{
    expand_indices, idim, idx_strides, reslice, reslice_mut, udim, DVector, DataType, Tensor,
};

macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width..][..$width]
    };
}

pub(super) fn gather<T, U>(x: &mut Tensor<T>, table: &Tensor<U>, tokens: &[utok])
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    debug_assert_eq!(x.data_type(), table.data_type());
    debug_assert_eq!(x.shape().last(), table.shape().last());

    let x = x.as_mut_slice();
    let table = table.as_slice();
    debug_assert_eq!(x.len() % tokens.len(), 0);

    let d = x.len() / tokens.len();
    for (i, &t) in tokens.iter().enumerate() {
        slice!(x; d; [i]).copy_from_slice(&slice!(table; d; [t]))
    }
}

pub(super) fn rms_norm<T, U, V>(o: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>, epsilon: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    let dt = o.data_type();
    debug_assert_eq!(x.data_type(), dt);
    debug_assert_eq!(w.data_type(), dt);
    debug_assert_eq!(o.shape(), x.shape());
    debug_assert_eq!(&[*o.shape().last().unwrap()], w.shape());

    let o = o.as_mut_slice();
    let x = x.as_slice();
    let w = w.as_slice();

    match dt {
        DataType::F16 => rms_norm_op(o, x, w, |x| {
            f16::from_f32(rms_norm_reduce(x.iter().copied().map(f16::to_f32), epsilon))
        }),
        DataType::F32 => rms_norm_op(o, x, w, |x| rms_norm_reduce(x.iter().copied(), epsilon)),
        _ => unreachable!("unsupported data type \"{dt:?}\""),
    }
}

pub(super) fn rms_norm_inplace<T, U>(o: &mut Tensor<T>, w: &Tensor<U>, epsilon: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let dt = o.data_type();
    debug_assert_eq!(w.data_type(), dt);
    debug_assert_eq!(&[*o.shape().last().unwrap()], w.shape());

    let o: &mut [u8] = o.as_mut_slice();
    let x = unsafe { std::slice::from_raw_parts(o.as_ptr(), o.len()) };
    let w = w.as_slice();

    match dt {
        DataType::F16 => rms_norm_op(o, x, w, |x| {
            f16::from_f32(rms_norm_reduce(x.iter().copied().map(f16::to_f32), epsilon))
        }),
        DataType::F32 => rms_norm_op(o, x, w, |x| rms_norm_reduce(x.iter().copied(), epsilon)),
        _ => unreachable!("unsupported data type \"{dt:?}\""),
    }
}

fn rms_norm_op<T: Copy + Mul<Output = T> + MulAssign>(
    o: &mut [u8],
    x: &[u8],
    w: &[u8],
    reduce: impl Fn(&[T]) -> T,
) {
    let o: &mut [T] = reslice_mut(o);
    let x: &[T] = reslice(x);
    let w: &[T] = reslice(w);
    let d = w.len();

    for i in 0..x.len() / w.len() {
        let o = &mut slice!(o; d; [i]);
        let x = &slice!(x; d; [i]);
        let k = reduce(x);
        zip(o, zip(x, w)).for_each(|(o, (x, w))| *o = *w * (k * *x));
    }
}

#[inline]
fn rms_norm_reduce(x: impl Iterator<Item = f32>, epsilon: f32) -> f32 {
    // (Σx^2 / n + δ)^(-1/2)
    let mut len = 0usize;
    let mut sum = 0.0f32;
    for x in x {
        len += 1;
        sum += x * x;
    }
    (sum / (len as f32) + epsilon).sqrt().recip()
}

/// c = a x b
///
/// - c: [N0, N1, ... , N_, m, n]
/// - a: [N0, N1, ... , N_, m, k]
/// - b: [N0, N1, ... , N_, k, n]
pub(super) fn mat_mul<T, U, V>(
    c: &mut Tensor<T>,
    beta: f32,
    a: &Tensor<U>,
    b: &Tensor<V>,
    alpha: f32,
) where
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

/// - t:   [N0, N1, ... , N_, num_head, head_dim]
/// - pos: [N0, N1, ... , N_]
pub(super) fn rotary_embedding<T, U>(t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    assert!(t.contiguous_len() >= 2);
    let [batch @ .., nh, dh] = t.shape() else {
        panic!()
    };
    assert_eq!(pos.shape(), batch);
    let nh = *nh as usize;
    let dh = *dh as usize / 2;

    let (n, idx_strides) = idx_strides(batch);
    for i in 0..n {
        let pos = pos
            .locate(&expand_indices(i, &idx_strides, &[1]).as_view())
            .unwrap()
            .cast::<udim>();
        let pos = unsafe { *pos } as f32;
        let ptr = t
            .locate_mut(&expand_indices(i, &idx_strides, &[0, 0, 1]).as_view())
            .unwrap()
            .cast::<(f16, f16)>();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, nh * dh) };
        for j in 0..nh {
            for (k, slice) in slice!(slice; dh ; [j]).iter_mut().enumerate() {
                let freq = pos / theta.powf(k as f32 / dh as f32);
                let (sin, cos) = freq.sin_cos();
                let (a, b) = slice;
                let a_ = a.to_f32();
                let b_ = b.to_f32();
                *a = f16::from_f32(a_ * cos - b_ * sin);
                *b = f16::from_f32(a_ * sin + b_ * cos);
            }
        }
    }
}

/// - x: [N0, N1, ... , N_, seq_len, att_len]
pub(super) fn softmax<T>(x: &mut Tensor<T>)
where
    T: DerefMut<Target = [u8]>,
{
    assert!(x.contiguous_len() >= 2);
    let (batch, dim) = x.shape().split_at(x.shape().len() - 2);
    let seq_len = dim[0] as usize;
    let att_len = dim[1] as usize;

    let (n, idx_strides) = idx_strides(batch);
    for i in 0..n {
        let ptr = x
            .locate_mut(&expand_indices(i, &idx_strides, &[0, 0, 1]).as_view())
            .unwrap()
            .cast::<f16>();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, seq_len * att_len) };
        for r in 0..seq_len {
            let slice = &mut slice!(slice; att_len; [r]);
            let (att, tail) = slice.split_at_mut(att_len - seq_len + r + 1);

            let max = att
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .to_f32();
            let sum = att
                .iter_mut()
                .map(|x| {
                    let exp = (x.to_f32() - max).exp();
                    *x = f16::from_f32(exp);
                    exp
                })
                .sum::<f32>();
            let sum = f16::from_f32(sum);
            att.iter_mut().for_each(|x| *x /= sum);

            tail.fill(f16::ZERO);
        }
    }
}

pub(super) fn swiglu<T, U>(gate: &mut Tensor<T>, up: &Tensor<U>)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let &[seq_len, di] = gate.shape() else {
        panic!("gate shape: {:?}", gate.shape());
    };
    assert_eq!(gate.data_type(), up.data_type());
    assert_eq!(up.shape(), &[seq_len, di]);
    assert!(gate.contiguous_len() >= 1);
    assert!(up.contiguous_len() >= 1);

    for i in 0..seq_len {
        let indices = DVector::from_vec(vec![i as idim, 0, 1]);
        let gate = gate.locate_mut(&indices.as_view()).unwrap();
        let gate = unsafe { std::slice::from_raw_parts_mut(gate.cast::<f16>(), di as usize) };
        let up = up.locate(&indices.as_view()).unwrap();
        let up = unsafe { std::slice::from_raw_parts(up.cast::<f16>(), di as usize) };
        for (gate, up) in gate.iter_mut().zip(up) {
            let x = gate.to_f32();
            let y = up.to_f32();

            #[inline(always)]
            fn sigmoid(x: f32) -> f32 {
                1. / (1. + (-x).exp())
            }

            *gate = f16::from_f32(x * sigmoid(x) * y);
        }
    }
}
