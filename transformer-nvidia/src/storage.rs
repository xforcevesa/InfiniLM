use cuda::{DevMem, DevSlice, Stream};
use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};
use tensor::Splitable;

pub struct Storage<'ctx>(Rc<DevMem<'ctx>>);

impl<'ctx> Storage<'ctx> {
    #[inline]
    pub fn new(size: usize, stream: &Stream<'ctx>) -> Self {
        Self(Rc::new(stream.malloc::<u8>(size)))
    }

    #[inline]
    pub unsafe fn borrow(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'ctx> From<DevMem<'ctx>> for Storage<'ctx> {
    #[inline]
    fn from(mem: DevMem<'ctx>) -> Self {
        Self(Rc::new(mem))
    }
}

impl<'ctx> Deref for Storage<'ctx> {
    type Target = DevSlice;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ctx> DerefMut for Storage<'ctx> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.get_mut() }
    }
}

impl<'ctx> Splitable for Storage<'ctx> {
    #[inline]
    fn split(&self) -> Self {
        Self(self.0.clone())
    }
}
