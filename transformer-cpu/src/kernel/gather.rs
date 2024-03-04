use super::slice;
use crate::Request;
use std::ops::{Deref, DerefMut};
use tensor::{udim, Tensor};

pub fn gather<T, U>(mut x: Tensor<T>, table: &Tensor<U>, requests: &[Request])
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let &[num_token, d] = x.shape() else { panic!() };

    debug_assert_eq!(x.data_type(), table.data_type());
    debug_assert_eq!(table.shape().len(), 2);
    debug_assert_eq!(table.shape()[1], d);
    debug_assert_eq!(
        requests.iter().map(Request::seq_len).sum::<udim>(),
        num_token
    );
    debug_assert!(x.is_contiguous());
    debug_assert!(table.is_contiguous());
    let d = d as usize * x.data_type().size();

    let x = x.as_mut_slice();
    let table = table.as_slice();
    for (i, &t) in requests.iter().flat_map(|s| s.tokens().iter()).enumerate() {
        slice!(x; d; [i]).copy_from_slice(&slice!(table; d; [t]))
    }
}
