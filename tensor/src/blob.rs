use crate::Splitable;
use std::{
    alloc::{alloc, dealloc, Layout},
    mem::align_of,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Blob {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}

impl Blob {
    #[inline]
    pub fn new(size: usize) -> Self {
        const ALIGN: usize = align_of::<usize>();
        let layout = Layout::from_size_align(size, ALIGN).unwrap();
        Self {
            ptr: NonNull::new(unsafe { alloc(layout) }).unwrap(),
            len: size,
        }
    }
}

impl Drop for Blob {
    #[inline]
    fn drop(&mut self) {
        const ALIGN: usize = align_of::<usize>();
        let layout = Layout::from_size_align(self.len, ALIGN).unwrap();
        unsafe { dealloc(self.ptr.as_ptr(), layout) }
    }
}

impl Deref for Blob {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Blob {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[repr(transparent)]
pub struct SplitableBlob(Arc<Blob>);

impl SplitableBlob {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self(Arc::new(Blob::new(size)))
    }
}

impl Splitable for SplitableBlob {
    #[inline]
    fn split(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Deref for SplitableBlob {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.0.ptr.as_ptr(), self.0.len) }
    }
}

impl DerefMut for SplitableBlob {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.0.ptr.as_ptr(), self.0.len) }
    }
}
