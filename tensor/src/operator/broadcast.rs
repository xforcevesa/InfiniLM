use super::Operator;
use crate::{udim, Affine, Shape};

pub struct Broadcast {
    shape: Shape,
}

impl Operator for Broadcast {
    #[inline]
    fn infer_shape(&self, input: &[udim]) -> Shape {
        debug_assert!(self.check_input(input));
        self.shape.clone()
    }

    fn to_affine(&self, input: &[udim]) -> Affine {
        debug_assert!(self.check_input(input));
        let nrows = self.shape.len() + 1;
        let ncols = input.len() + 1;
        let prefix = self.shape.len() - input.len();
        Affine::from_fn(nrows, ncols, |r, c| {
            if let Some(r) = r.checked_sub(prefix) {
                match input.get(c) {
                    Some(1) => 0,
                    _ if r == c => 1,
                    _ => 0,
                }
            } else {
                0
            }
        })
    }
}

impl Broadcast {
    fn check_input(&self, input: &[udim]) -> bool {
        input.len() <= self.shape.len()
            && input
                .iter()
                .rev()
                .zip(self.shape.iter().rev())
                .all(|(&i, &o)| i == 1 || i == o)
    }
}

#[test]
fn test() {
    let operator = Broadcast {
        shape: Shape::from_slice(&[2, 3, 4, 5]),
    };
    assert_eq!(
        operator.infer_shape(&[3, 1, 5]),
        Shape::from_slice(&[2, 3, 4, 5])
    );
    assert_eq!(
        operator.to_affine(&[3, 1, 5]),
        Affine::from_vec(
            5,
            4,
            vec![
                // column major
                0, 1, 0, 0, 0, //
                0, 0, 0, 0, 0, //
                0, 0, 0, 1, 0, //
                0, 0, 0, 0, 1, //
            ]
        )
    );
}
