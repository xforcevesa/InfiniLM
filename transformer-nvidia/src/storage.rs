use cuda::{DevMem, Stream};
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::{Deref, DerefMut},
    rc::Rc,
};
use tensor::PhysicalCell;

#[derive(Clone)]
pub struct Storage<'ctx>(Rc<RefCell<DevMem<'ctx>>>);
pub struct MemRef<'a, 'ctx: 'a>(Ref<'a, DevMem<'ctx>>);
pub struct MemRefMut<'a, 'ctx: 'a>(RefMut<'a, DevMem<'ctx>>);

impl<'ctx> PhysicalCell for Storage<'ctx> {
    type Raw = DevMem<'ctx>;
    type Access<'a> = MemRef<'a, 'ctx> where Self: 'a;
    type AccessMut<'a> = MemRefMut<'a, 'ctx> where Self: 'a;

    #[inline]
    unsafe fn get_unchecked(&self) -> &Self::Raw {
        &*self.0.as_ptr()
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self) -> &mut Self::Raw {
        &mut *self.0.as_ptr()
    }

    #[inline]
    fn access(&self) -> Self::Access<'_> {
        MemRef(self.0.borrow())
    }

    #[inline]
    fn access_mut(&mut self) -> Self::AccessMut<'_> {
        MemRefMut(self.0.borrow_mut())
    }
}

impl<'ctx> Deref for MemRef<'_, 'ctx> {
    type Target = DevMem<'ctx>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ctx> Deref for MemRefMut<'_, 'ctx> {
    type Target = DevMem<'ctx>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ctx> DerefMut for MemRefMut<'_, 'ctx> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'ctx> Storage<'ctx> {
    #[inline]
    pub fn new(len: usize, stream: &Stream<'ctx>) -> Self {
        Self(Rc::new(RefCell::new(stream.malloc::<u8>(len))))
    }
}

impl<'ctx> From<DevMem<'ctx>> for Storage<'ctx> {
    #[inline]
    fn from(value: DevMem<'ctx>) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
}
