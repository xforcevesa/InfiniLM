use crate::Request;
use tensor::udim;

#[inline]
pub fn pos<T, U>(requests: &[Request<T, U>], nt_hint: udim) -> Vec<u32> {
    let mut ans = Vec::<u32>::with_capacity(nt_hint as usize);
    for request in requests.iter() {
        ans.extend(request.pos()..request.att_len());
    }
    ans
}
