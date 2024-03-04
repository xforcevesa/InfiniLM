use super::slice;
use gemm::f16;
use std::ops::DerefMut;
use tensor::{expand_indices, idx_strides, Tensor};

/// - x: [N0, N1, ... , N_, seq_len, att_len]
pub fn softmax<T>(mut x: Tensor<T>)
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
