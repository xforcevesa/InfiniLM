use crate::{pattern::Pattern, udim, Affine, Shape, Tensor};

impl<Physical> Tensor<Physical> {
    pub fn transpose(self, perm: &[usize]) -> Self {
        let (shape, affine) = build(perm, &self.shape);

        Self {
            shape,
            pattern: Pattern(affine * self.pattern.0),
            ..self
        }
    }
}

fn build(perm: &[usize], input: &[udim]) -> (Shape, Affine) {
    let n = perm.len();
    assert_eq!(input.len(), n);
    let shape = perm.iter().map(|&i| input[i]).collect();
    let affine = Affine::from_fn(n + 1, n + 1, |r, c| {
        if c == perm.get(r).copied().unwrap_or(r) {
            1
        } else {
            0
        }
    });
    (shape, affine)
}

#[test]
fn test() {
    let (shape, affine) = build(&[0, 3, 1, 2], &[1, 2, 3, 4]);
    assert_eq!(shape.as_slice(), &[1, 4, 2, 3]);
    assert_eq!(
        affine.as_slice(),
        &[
            // column major
            1, 0, 0, 0, 0, //
            0, 0, 1, 0, 0, //
            0, 0, 0, 1, 0, //
            0, 1, 0, 0, 0, //
            0, 0, 0, 0, 1, //
        ]
    );
}
