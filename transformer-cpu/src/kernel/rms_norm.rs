use super::slice;
use gemm::f16;
use std::{
    iter::zip,
    ops::{Deref, DerefMut, Mul, MulAssign},
};
use tensor::{reslice, reslice_mut, DataType, Tensor};

pub fn rms_norm<T, U, V>(o: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>, epsilon: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    let &[.., d] = o.shape() else { panic!() };
    let dt = o.data_type();

    debug_assert_eq!(x.data_type(), dt);
    debug_assert_eq!(w.data_type(), dt);
    debug_assert_eq!(o.shape(), x.shape());
    debug_assert_eq!(w.shape(), &[d]);
    debug_assert!(o.is_contiguous());
    debug_assert!(x.is_contiguous());
    debug_assert!(w.is_contiguous());

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
