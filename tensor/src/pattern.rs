use crate::{idim, udim};
use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub(crate) struct Pattern(pub DVector<idim>);

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

pub type Shape = SmallVec<[udim; 4]>;
pub type Affine = DMatrix<idim>;

pub fn idx_strides(shape: &[udim]) -> (udim, Vec<udim>) {
    let mut idx_strides = vec![0; shape.len()];
    idx_strides[shape.len() - 1] = 1;
    for i in (1..shape.len()).rev() {
        idx_strides[i - 1] = idx_strides[i] * shape[i];
    }
    (shape[0] * idx_strides[0], idx_strides)
}

pub fn expand_indices(i: udim, idx_strides: &[udim], tail: &[idim]) -> DVector<idim> {
    let mut rem = i as idim;
    let mut ans = vec![0 as idim; idx_strides.len() + tail.len()];
    for (i, &s) in idx_strides.iter().enumerate() {
        ans[i] = rem / s as idim;
        rem %= s as idim;
    }
    ans[idx_strides.len()..].copy_from_slice(tail);
    DVector::from_vec(ans)
}
