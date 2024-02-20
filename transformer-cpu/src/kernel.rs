use common::{slice, utok};
use gemm::f16;
use std::{
    iter::zip,
    ops::{Mul, MulAssign},
};
use tensor::{DataType, Tensor};

pub(super) fn gather<T, U>(x: &mut Tensor<T>, table: &Tensor<U>, tokens: &[utok])
where
    T: AsMut<[u8]>,
    U: AsRef<[u8]>,
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

pub(super) fn rms_norm<T, U, V>(o: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>, theta: f32)
where
    T: AsMut<[u8]>,
    U: AsRef<[u8]>,
    V: AsRef<[u8]>,
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
            f16::from_f32(rms_norm_reduce(x.iter().copied().map(f16::to_f32), theta))
        }),
        DataType::F32 => op(o, x, w, |x| rms_norm_reduce(x.iter().copied(), theta)),
        _ => unreachable!(),
    }
}

#[inline]
fn rms_norm_reduce(x: impl Iterator<Item = f32>, theta: f32) -> f32 {
    // (Σx^2 / n + δ)^(-1/2)
    let mut len = 0usize;
    let mut sum = 0.0f32;
    for x in x {
        len += 1;
        sum += x * x;
    }
    (sum / (len as f32) + theta).sqrt().recip()
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
