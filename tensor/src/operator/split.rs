use super::Operator;
use crate::{idim, udim, Affine, Shape};
use smallvec::SmallVec;

pub struct Split {
    axis: udim,
    segments: Shape,
}

impl Operator for Split {
    fn build(&self, input: &[udim]) -> SmallVec<[(Shape, Affine); 1]> {
        debug_assert!(self.axis < input.len() as udim);
        debug_assert_eq!(input[self.axis as usize], self.segments.iter().sum());

        let n = input.len();
        let axis = self.axis as usize;
        self.segments
            .iter()
            .scan(0, |prefix, &seg| {
                let shape = input
                    .iter()
                    .enumerate()
                    .map(|(i, &dim)| if i == self.axis as usize { seg } else { dim })
                    .collect();
                let affine = Affine::from_fn(n + 1, n + 1, |r, c| {
                    if r == c {
                        1
                    } else if r == n {
                        if c == axis {
                            *prefix
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                });
                *prefix += seg as idim;
                Some((shape, affine))
            })
            .collect()
    }
}

#[test]
fn test() {
    let ans = Split {
        axis: 1,
        segments: Shape::from_slice(&[3, 4, 5]),
    }
    .build(&[11, 12, 13]);
    assert_eq!(ans.len(), 3);
    assert_eq!(ans[0].0.as_slice(), &[11, 3, 13]);
    assert_eq!(ans[1].0.as_slice(), &[11, 4, 13]);
    assert_eq!(ans[2].0.as_slice(), &[11, 5, 13]);
    assert_eq!(
        ans[0].1.as_slice(),
        &[
            // column major
            1, 0, 0, 0, //
            0, 1, 0, 0, //
            0, 0, 1, 0, //
            0, 0, 0, 1, //
        ]
    );
    assert_eq!(
        ans[1].1.as_slice(),
        &[
            // column major
            1, 0, 0, 0, //
            0, 1, 0, 3, //
            0, 0, 1, 0, //
            0, 0, 0, 1, //
        ]
    );
    assert_eq!(
        ans[2].1.as_slice(),
        &[
            // column major
            1, 0, 0, 0, //
            0, 1, 0, 7, //
            0, 0, 1, 0, //
            0, 0, 0, 1, //
        ]
    );
}
