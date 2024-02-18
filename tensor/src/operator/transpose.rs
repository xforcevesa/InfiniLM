use super::Operator;
use crate::{udim, Affine, Shape};
use smallvec::SmallVec;

pub struct Transpose {
    perm: SmallVec<[udim; 4]>,
}

impl Operator for Transpose {
    #[inline]
    fn infer_shape(&self, input: &[udim]) -> Shape {
        debug_assert_eq!(input.len(), self.perm.len());
        self.perm.iter().map(|&i| input[i as usize]).collect()
    }

    fn to_affine(&self, input: &[udim]) -> Affine {
        debug_assert_eq!(input.len(), self.perm.len());
        let n = self.perm.len();
        Affine::from_fn(n + 1, n + 1, |r, c| {
            if c == self.perm.get(r).map_or(r, |&p| p as usize) {
                1
            } else {
                0
            }
        })
    }
}

#[test]
fn test() {
    let operator = Transpose {
        perm: Shape::from_slice(&[0, 2, 1, 3]),
    };
    assert_eq!(
        operator.infer_shape(&[1, 2, 3, 4]),
        Shape::from_slice(&[1, 3, 2, 4])
    );
    assert_eq!(
        operator.to_affine(&[1, 2, 3, 4]),
        Affine::from_vec(
            5,
            5,
            vec![
                1, 0, 0, 0, 0, //
                0, 0, 1, 0, 0, //
                0, 1, 0, 0, 0, //
                0, 0, 0, 1, 0, //
                0, 0, 0, 0, 1, //
            ]
        )
    );
}
