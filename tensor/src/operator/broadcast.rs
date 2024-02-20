use super::Operator;
use crate::{udim, Affine, Shape};
use smallvec::SmallVec;

#[repr(transparent)]
pub struct Broadcast(Shape);

impl Operator for Broadcast {
    fn build(&self, input: &[udim]) -> SmallVec<[(Shape, Affine); 1]> {
        debug_assert!(self.check_input(input));
        let nrows = self.0.len() + 1;
        let ncols = input.len() + 1;
        let prefix = self.0.len() - input.len();
        let affine = Affine::from_fn(nrows, ncols, |r, c| {
            if let Some(r) = r.checked_sub(prefix) {
                match input.get(c) {
                    Some(1) => 0,
                    _ if r == c => 1,
                    _ => 0,
                }
            } else {
                0
            }
        });
        smallvec::smallvec![(self.0.clone(), affine)]
    }
}

impl Broadcast {
    fn check_input(&self, input: &[udim]) -> bool {
        input.len() <= self.0.len()
            && input
                .iter()
                .rev()
                .zip(self.0.iter().rev())
                .all(|(&i, &o)| i == 1 || i == o)
    }
}

#[test]
fn test() {
    let ans = Broadcast(Shape::from_slice(&[2, 3, 4, 5])).build(&[3, 1, 5]);
    assert_eq!(ans.len(), 1);
    assert_eq!(ans[0].0, Shape::from_slice(&[2, 3, 4, 5]));
    assert_eq!(
        ans[0].1,
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
