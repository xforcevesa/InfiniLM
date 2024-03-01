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
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.get_unchecked(),
        }
    }

    #[inline]
    pub unsafe fn access_unchecked_mut(&mut self) -> Tensor<&mut Physical::Raw> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.get_unchecked_mut(),
        }
    }

    #[inline]
    pub fn access(&self) -> Tensor<Physical::Access<'_>> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.access(),
        }
    }

    #[inline]
    pub fn access_mut(&mut self) -> Tensor<Physical::AccessMut<'_>> {
        Tensor {
            data_type: self.data_type,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: self.physical.access_mut(),
        }
    }
}
