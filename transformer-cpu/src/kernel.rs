use common::utok;
use gemm::{f16, gemm};
use std::{
    iter::zip,
    ops::{Deref, DerefMut, Mul, MulAssign},
};
use tensor::{expand_indices, idx_strides, reslice, reslice_mut, udim, DataType, Tensor};

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
                    gemm::Parallelism::None,
                )
            },
            _ => unreachable!(),
        }
    }
}

pub(super) fn rotary_embedding<T, U>(t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    assert!(t.contiguous_len() >= 2);
    let (batch, dim) = t.shape().split_at(t.shape().len() - 2);
    assert_eq!(batch, pos.shape());
    let nh = dim[0] as usize; // n heads
    let hd = dim[1] as usize; // head dim

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
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, nh * hd / 2) };
        for j in 0..nh {
            for (k, slice) in slice!(slice; hd / 2; [j]).iter_mut().enumerate() {
                let freq = pos / theta.powf(k as f32 * 2. / hd as f32);
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

pub(super) fn softmax<T>(_x: &mut Tensor<T>)
where
    T: DerefMut<Target = [u8]>,
{
    // TODO
}
