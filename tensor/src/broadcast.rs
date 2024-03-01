use crate::{pattern::Pattern, udim, Affine, Shape, Tensor};
use std::iter::zip;

impl<Physical> Tensor<Physical> {
    pub fn broadcast(self, shape: &[udim]) -> Self {
        Self {
            shape: Shape::from_slice(shape),
            pattern: Pattern(build(shape, &self.shape) * self.pattern.0),
            ..self
        }
    }
}

fn build(dst: &[udim], src: &[udim]) -> Affine {
    assert!(dst.len() >= src.len());
    assert!(zip(src.iter().rev(), dst.iter().rev()).all(|(&i, &o)| i == 1 || i == o));

    let nrows = dst.len() + 1;
    let ncols = src.len() + 1;
    let prefix = nrows - ncols;
    Affine::from_fn(nrows, ncols, |r, c| {
        if let Some(r) = r.checked_sub(prefix) {
            match src.get(c) {
                Some(1) => 0,
                _ if r == c => 1,
                _ => 0,
            }
        } else {
            0
        }
    })
}

#[test]
fn test() {
    assert_eq!(
        build(&[2, 3, 4, 5], &[3, 1, 5]).as_slice(),
        &[
            // column major
            0, 1, 0, 0, 0, //
            0, 0, 0, 0, 0, //
            0, 0, 0, 1, 0, //
            0, 0, 0, 0, 1, //
        ]
    );
}
