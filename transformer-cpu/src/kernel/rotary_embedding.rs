use super::slice;
use gemm::f16;
use std::ops::{Deref, DerefMut};
use tensor::{expand_indices, idx_strides, udim, DataType, Tensor};
use transformer::BetweenF32;

/// - t:   [num_token, num_head, head_dim]
/// - pos: [num_token]
pub fn rotary_embedding<T, U>(t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let &[nt, _, _] = t.shape() else { panic!() };
    assert_eq!(pos.data_type(), DataType::U32);
    assert_eq!(pos.shape(), &[nt]);
    assert!(t.contiguous_len() >= 2);

    let (n, idx_strides) = idx_strides(&[nt]);
    for i in 0..n {
        let pos = pos
            .locate(&expand_indices(i, &idx_strides, &[1]).as_view())
            .unwrap()
            .cast::<udim>();
        let pos = unsafe { *pos } as f32;
        match t.data_type() {
            DataType::F16 => typed::<T, f16>(t, i, &idx_strides, pos, theta),
            DataType::F32 => typed::<T, f32>(t, i, &idx_strides, pos, theta),
            _ => unreachable!(),
        }
    }
}

fn typed<T, U>(t: &mut Tensor<T>, i: udim, idx_strides: &[udim], pos: f32, theta: f32)
where
    T: DerefMut<Target = [u8]>,
    U: BetweenF32,
{
    let nh = t.shape()[1] as usize;
    let dh = t.shape()[2] as usize / 2;

    let ptr = t
        .locate_mut(&expand_indices(i, idx_strides, &[0, 0, 1]).as_view())
        .unwrap()
        .cast::<(U, U)>();
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, nh * dh) };
    for j in 0..nh {
        for (k, slice) in slice!(slice; dh ; [j]).iter_mut().enumerate() {
            let freq = pos / theta.powf(k as f32 / dh as f32);
            let (sin, cos) = freq.sin_cos();
            let (a, b) = slice;
            let a_ = a.get();
            let b_ = b.get();
            *a = U::cast(a_ * cos - b_ * sin);
            *b = U::cast(a_ * sin + b_ * cos);
        }
    }
}
