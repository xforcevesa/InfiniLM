use crate::{idim, udim, DataType, Operator};
use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;

pub struct Tensor<Physical> {
    data_type: DataType,
    shape: Shape,
    pattern: Pattern,
    physical: SmallVec<[Physical; 1]>,
}

impl<Physical: Clone> Tensor<Physical> {
    #[inline]
    pub fn new(
        data_type: DataType,
        shape: Shape,
        pattern: Pattern,
        physical: impl IntoIterator<Item = Physical>,
    ) -> Self {
        Self {
            data_type,
            shape,
            pattern,
            physical: physical.into_iter().collect(),
        }
    }

    #[inline]
    pub fn apply(&self, operator: &impl Operator) -> SmallVec<[Self; 1]> {
        operator
            .build(&self.shape)
            .into_iter()
            .map(|(shape, affine)| Self {
                data_type: self.data_type,
                shape,
                pattern: Pattern(affine * &self.pattern.0),
                physical: self.physical.clone(),
            })
            .collect()
    }
}

pub type Shape = SmallVec<[udim; 4]>;
pub type Affine = DMatrix<idim>;

pub struct Pattern(DVector<idim>);
