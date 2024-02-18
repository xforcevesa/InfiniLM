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
    pub fn apply(&self, operator: &impl Operator) -> Self {
        Self {
            data_type: self.data_type,
            shape: operator.infer_shape(&self.shape),
            pattern: Pattern(operator.to_affine(&self.shape) * &self.pattern.0),
            physical: self.physical.clone(),
        }
    }
}

pub type Shape = SmallVec<[udim; 4]>;
pub type Affine = DMatrix<idim>;

pub struct Pattern(DVector<idim>);
