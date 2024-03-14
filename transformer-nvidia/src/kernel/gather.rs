use common::utok;
use cuda::{bindings::CUdeviceptr, AsRaw, DevSlice, Stream};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn gather<'a, T, U, I>(x: Tensor<T>, table: &Tensor<U>, requests: I, stream: &Stream)
where
    T: DerefMut<Target = DevSlice>,
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

    let x = unsafe { x.physical().as_raw() };
    let table = table.as_slice();
    let stream = unsafe { stream.as_raw() };
    for (i, &t) in requests.into_iter().flatten().enumerate() {
        let src = table[d * t as usize..].as_ptr().cast();
        let dst = x + (d * i) as CUdeviceptr;
        cuda::driver!(cuMemcpyHtoDAsync_v2(dst, src, d, stream));
    }
}
