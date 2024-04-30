use std::{
    ops::{DerefMut, Range},
    slice::from_raw_parts,
};
use tensor::Tensor;

/// 解码的要求。
pub struct DecodingMeta {
    /// 查询的长度。
    pub num_query: usize,
    /// 解码的长度。
    pub num_decode: usize,
}

impl DecodingMeta {
    /// 根据解码元信息移动数据。
    pub fn select<T, B>(
        x: &mut Tensor<T>,
        decoding: impl IntoIterator<Item = Self>,
        mut mem_mov: impl FnMut(&mut [B], &[B]),
    ) -> Range<usize>
    where
        T: DerefMut<Target = [B]>,
    {
        let dt = x.data_type();
        let &[_nt, d] = x.shape() else {
            panic!("shape error")
        };
        let dst_ = &mut **x.physical_mut();
        let src_ = unsafe { from_raw_parts(dst_.as_ptr(), dst_.len()) };

        let mut iter = decoding.into_iter();
        let mut begin = 0;
        let mut src = 0;
        let mut dst = 0;
        for DecodingMeta {
            num_query,
            num_decode,
        } in iter.by_ref()
        {
            begin += num_query;
            if num_decode > 0 {
                src = begin;
                dst = begin;
                begin -= num_decode;
                break;
            }
        }
        let len = d as usize * dt.size();
        for DecodingMeta {
            num_query,
            num_decode,
        } in iter
        {
            src += num_query - num_decode;
            if src > dst {
                for _ in 0..num_decode {
                    mem_mov(&mut dst_[dst * len..][..len], &src_[src * len..][..len]);
                    src += 1;
                    dst += 1;
                }
            } else {
                src += num_decode;
                dst += num_decode;
            }
        }
        begin..dst
    }
}
