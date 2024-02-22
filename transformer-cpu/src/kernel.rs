use common::{upos, utok};
use gemm::{f16, gemm};
use std::{
    iter::zip,
    ops::{Deref, DerefMut, Mul, MulAssign},
};
use tensor::{expand_indices, idx_strides, udim, DataType, Tensor};

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

    fn op<T: Copy + Mul<Output = T> + MulAssign>(
        o: &mut [u8],
        x: &[u8],
        w: &[u8],
        reduce: impl Fn(&[T]) -> T,
    ) {
        let o: &mut [T] = unsafe { slice_as_mut(o) };
        let x: &[T] = unsafe { slice_as(x) };
        let w: &[T] = unsafe { slice_as(w) };
        let d = w.len();

        for i in 0..x.len() / w.len() {
            let o = &mut slice!(o; d; [i]);
            let x = &slice!(x; d; [i]);
            let k = reduce(x);
            zip(o, zip(x, w)).for_each(|(o, (x, w))| *o = *w * (k * *x));
        }
    }

    match dt {
        DataType::F16 => op(o, x, w, |x| {
            f16::from_f32(rms_norm_reduce(x.iter().copied().map(f16::to_f32), epsilon))
        }),
        DataType::F32 => op(o, x, w, |x| rms_norm_reduce(x.iter().copied(), epsilon)),
        _ => unreachable!("unsupported data type \"{dt:?}\""),
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

pub(super) fn matmul<T, U, V>(c: &mut Tensor<T>, a: &Tensor<U>, b: &Tensor<V>)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    let dt = c.data_type();
    assert_eq!(a.data_type(), dt);
    assert_eq!(b.data_type(), dt);
    assert_eq!(c.shape().len(), 2);
    assert_eq!(a.shape().len(), 2);
    assert_eq!(b.shape().len(), 2);
    let m = c.shape()[0];
    let n = c.shape()[1];
    let k = a.shape()[1];
    assert_eq!(a.shape(), &[m, k]);
    assert_eq!(b.shape(), &[k, n]);

    let dst = c.locate_start_mut();
    let dst_strides = c.strides();
    let dst_cs = dst_strides[1] as isize;
    let dst_rs = dst_strides[0] as isize;

    let lhs = a.as_ptr();
    let lhs_strides = a.strides();
    let lhs_cs = lhs_strides[1] as isize;
    let lhs_rs = lhs_strides[0] as isize;

    let rhs = b.as_ptr();
    let rhs_strides = b.strides();
    let rhs_cs = rhs_strides[1] as isize;
    let rhs_rs = rhs_strides[0] as isize;

    match dt {
        DataType::F32 => unsafe {
            gemm(
                m as _,
                n as _,
                k as _,
                dst.cast::<f32>(),
                dst_cs,
                dst_rs,
                false,
                lhs.cast::<f32>(),
                lhs_cs,
                lhs_rs,
                rhs.cast::<f32>(),
                rhs_cs,
                rhs_rs,
                0.,
                1.,
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
                false,
                lhs.cast::<f16>(),
                lhs_cs,
                lhs_rs,
                rhs.cast::<f16>(),
                rhs_cs,
                rhs_rs,
                f16::from_f32(0.),
                f16::from_f32(1.),
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            )
        },
        _ => unreachable!(),
    }
}

pub(super) fn rotary_embedding<T>(t: &mut Tensor<T>, head_dim: udim, pos: upos, theta: f32)
where
    T: DerefMut<Target = [u8]>,
{
    assert!(t.contiguous_len() > 0);
    let (len, batch) = t.shape().split_last().unwrap();
    let len = *len as usize / 2;
    let (n, idx_strides) = idx_strides(batch);
    let mul = 2. / head_dim as f32;
    for i in 0..n {
        let indices = expand_indices(i, &idx_strides, &[0, 1]);
        let pos = (pos + indices[indices.len() - 3] as upos) as f32;
        let ptr = t
            .locate_mut(&indices.as_view())
            .unwrap()
            .cast::<(f16, f16)>();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        for (j, (a, b)) in slice.iter_mut().enumerate() {
            let freq = pos / theta.powf((j as f32 * mul).fract());
            let (sin, cos) = freq.sin_cos();
            let a_ = a.to_f32();
            let b_ = b.to_f32();
            *a = f16::from_f32(a_ * cos - b_ * sin);
            *b = f16::from_f32(a_ * sin + b_ * cos);
        }
    }
}

#[inline(always)]
unsafe fn slice_as<T, U>(x: &[U]) -> &[T] {
    let (head, body, tail) = x.align_to::<T>();
    debug_assert!(head.is_empty());
    debug_assert!(tail.is_empty());
    body
}

#[inline(always)]
unsafe fn slice_as_mut<T, U>(x: &mut [U]) -> &mut [T] {
    let (head, body, tail) = x.align_to_mut::<T>();
    debug_assert!(head.is_empty());
    debug_assert!(tail.is_empty());
    body
}
