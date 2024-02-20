use super::Operator;
use crate::{idim, udim, Affine, Shape};
use smallvec::{smallvec, SmallVec};

#[repr(transparent)]
pub struct Squeeze(Vec<SqueezeOp>);

#[repr(u8)]
enum SqueezeOp {
    Insert,
    Remove,
    Skip,
}

impl Operator for Squeeze {
    fn build(&self, input: &[udim]) -> SmallVec<[(Shape, Affine); 1]> {
        let mut shape = Shape::new();
        let mut sources = Vec::<idim>::new();
        {
            let mut idx = 0;
            for op in &self.0 {
                match op {
                    SqueezeOp::Insert => {
                        shape.push(1);
                        sources.push(-1);
                    }
                    SqueezeOp::Remove => {
                        assert_eq!(input[idx], 1);
                        idx += 1;
                    }
                    SqueezeOp::Skip => {
                        shape.push(input[idx]);
                        sources.push(idx as idim);
                        idx += 1;
                    }
                }
            }
            assert_eq!(idx, input.len());
        }
        let nrows = shape.len() + 1;
        let ncols = input.len() + 1;
        let affine = Affine::from_fn(nrows, ncols, |r, c| {
            if sources.get(r).map_or(c == ncols - 1, |&s| s == c as idim) {
                1
            } else {
                0
            }
        });
        smallvec![(shape, affine)]
    }
}

#[test]
fn test() {
    let ans = Squeeze(vec![
        SqueezeOp::Remove,
        SqueezeOp::Skip,
        SqueezeOp::Insert,
        SqueezeOp::Skip,
        SqueezeOp::Insert,
        SqueezeOp::Skip,
    ])
    .build(&[1, 3, 224, 224]);
    assert_eq!(ans.len(), 1);
    assert_eq!(ans[0].0.as_slice(), &[3, 1, 224, 1, 224]);
    assert_eq!(
        ans[0].1.as_slice(),
        &[
            // column major
            0, 0, 0, 0, 0, 0, //
            1, 0, 0, 0, 0, 0, //
            0, 0, 1, 0, 0, 0, //
            0, 0, 0, 0, 1, 0, //
            0, 0, 0, 0, 0, 1, //
        ]
    );
}
