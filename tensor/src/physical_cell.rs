use crate::Tensor;
use std::ops::{Deref, DerefMut};

pub trait PhysicalCell {
    type Raw: ?Sized;
    type Access<'a>: Deref<Target = Self::Raw>
    where
        Self: 'a;
    type AccessMut<'a>: DerefMut<Target = Self::Raw>
    where
        Self: 'a;

    unsafe fn get_unchecked(&self) -> &Self::Raw;
    unsafe fn get_unchecked_mut(&mut self) -> &mut Self::Raw;
    fn access(&self) -> Self::Access<'_>;
    fn access_mut(&mut self) -> Self::AccessMut<'_>;
}

impl<Physical: PhysicalCell> Tensor<Physical> {
    #[inline]
    pub unsafe fn access_unchecked(&self) -> Tensor<&Physical::Raw> {
        self.as_ref()
            .map_physical(|physical| physical.get_unchecked())
    }

    #[inline]
    pub unsafe fn access_unchecked_mut(&mut self) -> Tensor<&mut Physical::Raw> {
        self.as_mut()
            .map_physical(|physical| physical.get_unchecked_mut())
    }

    #[inline]
    pub fn access(&self) -> Tensor<Physical::Access<'_>> {
        unsafe { self.as_ref().map_physical(|physical| physical.access()) }
    }

    #[inline]
    pub fn access_mut(&mut self) -> Tensor<Physical::AccessMut<'_>> {
        unsafe { self.as_mut().map_physical(|physical| physical.access_mut()) }
    }
}
