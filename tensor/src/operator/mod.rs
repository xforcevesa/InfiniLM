mod transpose;

use crate::{udim, Affine, Shape};

pub trait Operator {
    fn infer_shape(&self, input: &[udim]) -> Shape;
    fn to_affine(&self, input: &[udim]) -> Affine;
}
