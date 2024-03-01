use cuda::{Context, ContextGuard};
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
