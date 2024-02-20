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

impl<Physical> Tensor<Physical> {
    pub fn new(data_type: DataType, shape: &[usize], physical: Physical) -> Self {
        let shape = Shape::from_iter(shape.iter().map(|&d| d as udim));
        Self {
            data_type,
            pattern: Pattern::from_shape(&shape, 0),
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
    pub fn strides(&self) -> &[idim] {
        &self.pattern.strides()
    }

    #[inline]
    pub fn offset(&self) -> udim {
        self.pattern.offset() as _
    }

    #[inline]
    pub const fn physical(&self) -> &Physical {
        &self.physical
    }

    #[inline]
    pub fn physical_mut(&mut self) -> &mut Physical {
        &mut self.physical
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn is_contiguous(&self) -> bool {
        let strides = self.pattern.0.as_slice();
        let n = self.shape.len() - 1;
        strides[n] == 1 && (0..n).all(|i| strides[i] == strides[i + 1] * self.shape[i + 1] as idim)
    }

    #[inline]
    pub unsafe fn cast(&self, dtype: DataType, physical: Physical) -> Self {
        Self {
            data_type: dtype,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical,
        }
    }
}

impl<Physical: Clone> Tensor<Physical> {
    pub fn reshape(&self, shape: Shape) -> Self {
        assert!(self.is_contiguous());
        assert_eq!(
            self.shape.iter().product::<udim>(),
            shape.iter().product::<udim>(),
        );
        Self {
            data_type: self.data_type,
            pattern: Pattern::from_shape(&shape, self.pattern.offset()),
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

impl<Physical: AsRef<[u8]>> Tensor<Physical> {
    pub fn as_slice(&self) -> &[u8] {
        self.physical.as_ref()
    }
}

impl<Physical: AsMut<[u8]>> Tensor<Physical> {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.physical.as_mut()
    }
}

pub type Shape = SmallVec<[udim; 4]>;
pub type Affine = DMatrix<idim>;

#[derive(Clone, Debug)]
struct Pattern(DVector<idim>);

impl Pattern {
    pub fn from_shape(shape: &[udim], offset: idim) -> Self {
        let n = shape.len();
        let mut strides = vec![0; n + 1];
        strides[n - 1] = 1;
        strides[n] = offset;
        for i in (1..n).rev() {
            strides[i - 1] = strides[i] * shape[i] as idim;
        }
        Self(DVector::from_vec(strides))
    }

    #[inline]
    pub fn strides(&self) -> &[idim] {
        &self.0.as_slice()[..self.0.len() - 1]
    }

    #[inline]
    pub fn offset(&self) -> idim {
        self.0[self.0.len() - 1]
    }
}

#[test]
fn test() {
    use super::Transpose;

    let t = Tensor::new(DataType::F32, &[2, 3, 4, 5], ());
    assert_eq!(t.shape(), &[2, 3, 4, 5]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 5, 1, 0]);
    assert_eq!(t.is_contiguous(), true);

    let t = t.reshape(Shape::from_slice(&[2, 3, 20]));
    assert_eq!(t.shape(), &[2, 3, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 1, 0]);
    assert_eq!(t.is_contiguous(), true);

    let t = t.apply(&Transpose::new(&[2, 0, 1]));
    assert_eq!(t.len(), 1);
    let t = t.into_iter().next().unwrap();
    assert_eq!(t.shape(), &[20, 2, 3]);
    assert_eq!(t.pattern.0.as_slice(), &[1, 60, 20, 0]);
    assert_eq!(t.is_contiguous(), false);
}
