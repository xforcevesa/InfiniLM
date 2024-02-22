use std::{
    cell::{Ref, RefCell, RefMut},
    ops::{Deref, DerefMut},
    rc::Rc,
};

#[derive(Clone, Debug)]
pub struct Storage(Rc<RefCell<Vec<u8>>>);
pub struct VecRef<'a>(Ref<'a, Vec<u8>>);
pub struct VecRefMut<'a>(RefMut<'a, Vec<u8>>);

impl tensor::Storage for Storage {
    type Access<'a> = VecRef<'a>;
    type AccessMut<'a> = VecRefMut<'a>;

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

impl Storage {
    pub fn new(size: usize) -> Self {
        Self(Rc::new(RefCell::new(vec![0; size])))
    }
}
