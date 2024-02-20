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

    pub fn is_contiguous(&self) -> bool {
        let strides = self.pattern.0.as_slice();
        let n = strides.len();
        &strides[n - 2..] == &[1, 0]
            && (0..n - 2).all(|i| strides[i] == strides[i + 1] * self.shape[i + 1] as idim)
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        debug_assert!(self.is_contiguous());
        debug_assert_eq!(
            self.shape.iter().product::<udim>(),
            shape.iter().product::<udim>(),
        );
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

#[test]
fn test() {
    use super::Transpose;
    use smallvec::smallvec;

    let t = Tensor::new(DataType::F32, Shape::from_slice(&[2, 3, 4, 5]), ());
    assert_eq!(t.shape(), &[2, 3, 4, 5]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 5, 1, 0]);
    assert_eq!(t.is_contiguous(), true);

    let t = t.reshape(Shape::from_slice(&[2, 3, 20]));
    assert_eq!(t.shape(), &[2, 3, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 1, 0]);
    assert_eq!(t.is_contiguous(), true);

    let t = t.apply(&Transpose::new(smallvec![2, 0, 1]));
    assert_eq!(t.len(), 1);
    let t = t.into_iter().next().unwrap();
    assert_eq!(t.shape(), &[20, 2, 3]);
    assert_eq!(t.pattern.0.as_slice(), &[1, 60, 20, 0]);
    assert_eq!(t.is_contiguous(), false);
}
