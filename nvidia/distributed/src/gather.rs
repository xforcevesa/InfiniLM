use common_nv::{
    cuda::{ContextSpore, DevMemSpore, StreamSpore},
    utok, Tensor,
};
use std::ops::Deref;

pub fn gather<T, I>(
    x: &mut [Tensor<DevMemSpore>],
    table: &Tensor<T>,
    tokens: I,
    comms: &nccl::CommunicatorGroup,
    streams: &[StreamSpore],
) where
    T: Deref<Target = [u8]>,
    I: IntoIterator<Item = utok>,
{
    assert!(!x.is_empty());
    let dt = x[0].data_type();
    let &[t, d] = x[0].shape() else { panic!() };

    assert_eq!(t as usize % x.len(), 0);
    let distributed = t as usize / x.len();

    assert_eq!(x.len(), comms.len());
    assert_eq!(x.len(), streams.len());
    assert!(x
        .iter()
        .skip(1)
        .all(|x| x.data_type() == dt && x.shape() == [t, d]));
    assert!(x.iter().all(|x| x.is_contiguous()));

    assert_eq!(table.data_type(), dt);
    assert_eq!(table.shape().len(), 2);
    assert_eq!(table.shape()[1], d);
    assert!(table.is_contiguous());

    let d = d as usize * dt.size();
    let table = table.as_slice();
    let mut iter = tokens.into_iter().map(|t| t as usize).enumerate();
    for (i, comm) in comms.call().iter().enumerate() {
        comm.device().retain_primary().apply(|ctx| {
            let stream = unsafe { streams[i].sprout(ctx) };
            let mut dst = unsafe { x[i].physical_mut().sprout(ctx) };
            for _ in 0..distributed {
                let Some((i, t)) = iter.next() else { break };
                stream.memcpy_h2d(&mut dst[d * i..][..d], &table[d * t..][..d]);
            }
            comm.all_gather(&mut dst, None, &stream);
        });
    }
}
