use crate::{idim, udim, DataType, Operator};
use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct Tensor<Physical> {
    data_type: DataType,
    shape: Shape,
    pattern: Pattern,
    physical: Physical,
}

impl<Physical: Clone> Tensor<Physical> {
    #[inline]
    pub fn new(data_type: DataType, shape: Shape, physical: Physical) -> Self {
        Self {
            data_type,
            pattern: Pattern::from_shape(&shape),
            shape,
            physical,
        }
    }

    #[inline]
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }

    #[inline]
    pub fn shape(&self) -> &[udim] {
        &self.shape
    }

    #[inline]
    pub const fn physical(&self) -> &Physical {
        &self.physical
    }

    #[inline]
    pub fn reshape(&self, shape: Shape) -> Self {
        debug_assert_eq!(self.pattern.0, Pattern::from_shape(&self.shape).0);
        Self {
            data_type: self.data_type,
            pattern: Pattern::from_shape(&shape),
            shape,
            physical: self.physical.clone(),
        }
    }

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

#[derive(Clone, Debug)]
pub struct Pattern(DVector<idim>);

impl Pattern {
    pub fn from_shape(shape: &[udim]) -> Self {
        let n = shape.len();
        let mut strides = vec![0; n + 1];
        strides[n - 1] = 1;
        for i in (1..n).rev() {
            strides[i - 1] = strides[i] * shape[i] as idim;
        }
        Self(DVector::from_vec(strides))
    }
}
