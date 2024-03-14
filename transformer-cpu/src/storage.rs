use std::{
    alloc::{alloc, Layout},
    mem::align_of,
    ops::{Deref, DerefMut},
    rc::Rc,
};
use tensor::Splitable;

pub struct Storage(Rc<Internal>);

impl Storage {
    #[inline]
    pub fn new(size: usize) -> Self {
        const ALIGN: usize = align_of::<usize>();
        let layout = Layout::from_size_align(size, ALIGN).unwrap();
        Self(Rc::new(Internal {
            ptr: unsafe { alloc(layout) },
            len: size,
        }))
    }
}

impl Splitable for Storage {
    #[inline]
    fn split(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Deref for Storage {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.0.ptr, self.0.len) }
    }
}

impl DerefMut for Storage {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.0.ptr, self.0.len) }
    }
}

struct Internal {
    ptr: *mut u8,
    len: usize,
}

impl Drop for Internal {
    #[inline]
    fn drop(&mut self) {
        const ALIGN: usize = align_of::<usize>();
        let layout = Layout::from_size_align(self.len, ALIGN).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}
