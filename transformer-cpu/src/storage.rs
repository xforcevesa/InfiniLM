use std::{
    cell::{Ref, RefCell, RefMut},
    ops::{Deref, DerefMut},
    rc::Rc,
};
use tensor::PhysicalCell;

#[derive(Debug)]
#[repr(transparent)]
pub struct Unique(Vec<u8>);

impl Deref for Unique {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Unique {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Unique {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self(vec![0u8; size])
    }
}

#[derive(Clone, Debug)]
pub struct Cell(Rc<RefCell<Vec<u8>>>);
pub struct VecRef<'a>(Ref<'a, Vec<u8>>);
pub struct VecRefMut<'a>(RefMut<'a, Vec<u8>>);

impl PhysicalCell for Cell {
    type Raw = [u8];
    type Access<'a> = VecRef<'a>;
    type AccessMut<'a> = VecRefMut<'a>;

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
        VecRef(self.0.borrow())
    }

    #[inline]
    fn access_mut(&mut self) -> Self::AccessMut<'_> {
        VecRefMut(self.0.borrow_mut())
    }
}

impl Deref for VecRef<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for VecRefMut<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for VecRefMut<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Cell {
    pub fn new(size: usize) -> Self {
        Self(Rc::new(RefCell::new(vec![0; size])))
    }
}
