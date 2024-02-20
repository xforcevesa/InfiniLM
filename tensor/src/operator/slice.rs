use super::Operator;
use crate::{idim, udim, Affine, Shape};
use smallvec::{smallvec, SmallVec};

#[repr(transparent)]
pub struct Slice(Vec<SliceDim>);

pub struct SliceDim {
    pub start: udim,
    pub step: idim,
    pub len: udim,
}

impl Operator for Slice {
    fn build(&self, input: &[udim]) -> SmallVec<[(Shape, Affine); 1]> {
        debug_assert_eq!(input.len(), self.0.len());
        debug_assert!(self.0.iter().zip(input).all(|(d, &i)| {
            let range = 0..i as idim;
            let start = d.start as idim;
            let end = start + (d.len - 1) as idim * d.step;
            range.contains(&start) && range.contains(&end)
        }));

        let n = self.0.len();
        let affine = Affine::from_fn(n + 1, n + 1, |r, c| {
            if r == n {
                self.0.get(c).map_or(1, |d| d.start as _)
            } else if r == c {
                self.0.get(c).map(|d| d.step).unwrap()
            } else {
                0
            }
        });
        smallvec![(self.0.iter().map(|d| d.len).collect(), affine)]
    }
}

#[test]
fn test() {
    let ans = Slice(vec![
        SliceDim {
            start: 2,
            step: 1,
            len: 2,
        },
        SliceDim {
            start: 0,
            step: 1,
            len: 4,
        },
        SliceDim {
            start: 1,
            step: 2,
            len: 3,
        },
    ])
    .build(&[5, 6, 7]);
    assert_eq!(ans.len(), 1);
    assert_eq!(ans[0].0, Shape::from_slice(&[2, 4, 3]));
    assert_eq!(
        ans[0].1,
        Affine::from_vec(
            4,
            4,
            vec![
                // column major
                1, 0, 0, 2, //
                0, 1, 0, 0, //
                0, 0, 2, 1, //
                0, 0, 0, 1, //
            ]
        )
    );
}
