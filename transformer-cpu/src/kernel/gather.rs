use super::slice;
use common::utok;
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn gather<'a, T, U, I>(mut x: Tensor<T>, table: &Tensor<U>, requests: I)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    I: IntoIterator<Item = &'a [utok]>,
{
    let &[_, d] = x.shape() else { panic!() };

    debug_assert_eq!(x.data_type(), table.data_type());
    debug_assert_eq!(table.shape().len(), 2);
    debug_assert_eq!(table.shape()[1], d);
    debug_assert!(x.is_contiguous());
    debug_assert!(table.is_contiguous());
    let d = d as usize * x.data_type().size();

    let x = x.as_mut_slice();
    let table = table.as_slice();
    for (i, &t) in requests.into_iter().flatten().enumerate() {
        slice!(x; d; [i]).copy_from_slice(&slice!(table; d; [t]))
    }
}
