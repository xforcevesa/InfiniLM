use super::slice;
use common::utok;
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn gather<T, U>(mut x: Tensor<T>, table: &Tensor<U>, tokens: &[&[utok]])
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let &[num_token, d] = x.shape() else { panic!() };

    debug_assert_eq!(x.data_type(), table.data_type());
    debug_assert_eq!(table.shape().len(), 2);
    debug_assert_eq!(table.shape()[1], d);
    debug_assert_eq!(
        tokens.iter().map(|s| s.len()).sum::<usize>(),
        num_token as usize
    );
    debug_assert!(x.is_contiguous());
    debug_assert!(table.is_contiguous());
    let d = d as usize * x.data_type().size();

    let x = x.as_mut_slice();
    let table = table.as_slice();
    for (i, &t) in tokens.iter().flat_map(|s| s.iter()).enumerate() {
        slice!(x; d; [i]).copy_from_slice(&slice!(table; d; [t]))
    }
}
