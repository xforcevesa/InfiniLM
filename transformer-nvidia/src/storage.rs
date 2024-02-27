use cuda::{AsRaw, Context, ContextGuard, Stream};
use std::{
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

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr as _, self.len) }
    }
}

impl DerefMut for PageLockedMemory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as _, self.len) }
    }
}

#[derive(Clone)]
pub(crate) struct DevMem<'a> {
    ptr: u64,
    len: usize,
    _stream: &'a Stream<'a>,
}

impl<'a> DevMem<'a> {
    pub fn new(len: usize, stream: &'a Stream) -> Self {
        let mut ptr = 0;
        cuda::driver!(cuMemAllocAsync(&mut ptr, len, stream.as_raw()));
        Self {
            ptr: ptr as _,
            len,
            _stream: stream,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl AsRaw for DevMem<'_> {
    type Raw = cuda::bindings::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}
