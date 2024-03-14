use super::slice;
use gemm::f16;
use std::ops::{Deref, DerefMut};
use tensor::{expand_indices, idx_strides, udim, Tensor};

/// - t:   [num_token, num_head, head_dim]
/// - pos: [num_token]
pub fn rotary_embedding<T, U>(t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    assert!(t.contiguous_len() >= 2);
    let &[num_tokens, nh, dh] = t.shape() else {
        panic!()
    };
    assert_eq!(pos.shape(), &[num_tokens]);
    let nh = nh as usize;
    let dh = dh as usize / 2;

    let (n, idx_strides) = idx_strides(&[num_tokens]);
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
