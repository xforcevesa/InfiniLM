use cuda::{AsRaw, Context, ContextGuard, Stream};
use std::{
    mem::size_of_val,
    ops::{Deref, DerefMut},
    ptr::null_mut,
    sync::Arc,
};

pub struct PageLockedMemory {
    ptr: usize,
    len: usize,
    ctx: Arc<Context>,
}

impl PageLockedMemory {
    pub fn new(len: usize, ctx: &ContextGuard) -> Self {
        let mut ptr = null_mut();
        cuda::driver!(cuMemHostAlloc(&mut ptr, len, 0));
        Self {
            ptr: ptr as _,
            len,
            ctx: ctx.clone_ctx(),
        }
    }
}

impl Drop for PageLockedMemory {
    #[inline]
    fn drop(&mut self) {
        self.ctx
            .apply(|_| cuda::driver!(cuMemFreeHost(self.ptr as _)));
    }
}

impl Deref for PageLockedMemory {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr as _, self.len) }
    }
}

impl DerefMut for PageLockedMemory {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as _, self.len) }
    }
}

#[derive(Clone)]
pub struct DevMem<'a> {
    ptr: cuda::bindings::CUdeviceptr,
    len: usize,
    _stream: &'a Stream<'a>,
}

impl<'a> DevMem<'a> {
    pub fn new(len: usize, stream: &'a Stream) -> Self {
        let mut ptr = 0;
        cuda::driver!(cuMemAllocAsync(&mut ptr, len, stream.as_raw()));
        Self {
            ptr,
            len,
            _stream: stream,
        }
    }

    pub fn from_slice<T: Copy>(slice: &[T], stream: &'a Stream) -> Self {
        let stream_ = unsafe { stream.as_raw() };
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = 0;
        cuda::driver!(cuMemAllocAsync(&mut ptr, len, stream_));
        cuda::driver!(cuMemcpyHtoDAsync_v2(ptr, src, len, stream_));
        Self {
            ptr,
            len,
            _stream: stream,
        }
    }
}

impl DevMem<'_> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn copy_in<T: Copy>(&mut self, slice: &[T], stream: &Stream) {
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        assert_eq!(len, self.len);
        cuda::driver!(cuMemcpyHtoDAsync_v2(self.ptr, src, len, stream.as_raw()));
    }
}

impl AsRaw for DevMem<'_> {
    type Raw = cuda::bindings::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}
