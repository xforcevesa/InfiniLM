use common::{slice, utok};
use gemm::f16;
use model_parameters::DataType;
use std::{
    iter::zip,
    ops::{Mul, MulAssign},
};

pub(super) fn gather(x: &mut [u8], table: &[u8], tokens: &[utok]) {
    debug_assert_eq!(x.len() % tokens.len(), 0);

    let d = x.len() / tokens.len();
    for (i, &t) in tokens.iter().enumerate() {
        slice!(x; d; [i]).copy_from_slice(&slice!(table; d; [t]))
    }
}

pub(super) fn rms_norm(o: &mut [u8], x: &[u8], w: &[u8], theta: f32, dt: DataType) {
    debug_assert_eq!(o.len(), x.len());

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
        DataType::BF16 => unreachable!(),
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
