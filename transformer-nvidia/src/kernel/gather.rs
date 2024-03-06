use common::utok;
use cuda::{bindings::CUdeviceptr, AsRaw, DevMem, Stream};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn gather<'a, T, U>(x: &mut Tensor<T>, table: &Tensor<U>, tokens: &[utok], stream: &Stream)
where
    T: DerefMut<Target = DevMem<'a>>,
    U: Deref<Target = [u8]>,
{
    debug_assert_eq!(x.data_type(), table.data_type());
    debug_assert_eq!(x.shape().last(), table.shape().last());

    let x_len = x.physical().len();
    let x = unsafe { x.physical().as_raw() };
    let table = table.as_slice();
    debug_assert_eq!(x_len % tokens.len(), 0);

    let d = x_len / tokens.len();
    let stream = unsafe { stream.as_raw() };
    for (i, &t) in tokens.iter().enumerate() {
        let src = table[d * t as usize..].as_ptr().cast();
        let dst = x + (d * i) as CUdeviceptr;
        cuda::driver!(cuMemcpyHtoDAsync_v2(dst, src, d, stream));
    }
}
