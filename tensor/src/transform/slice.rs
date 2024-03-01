use crate::{idim, pattern::Pattern, udim, Affine, Shape, Tensor};

impl<Physical> Tensor<Physical> {
    pub fn slice(self, dims: &[SliceDim]) -> Self {
        let (shape, affine) = build(dims, &self.shape);
        Self {
            shape,
            pattern: Pattern(affine * self.pattern.0),
            ..self
        }
    }
}

#[derive(Clone, Debug)]
pub struct SliceDim {
    pub start: udim,
    pub step: idim,
    pub len: udim,
}

#[macro_export]
macro_rules! slice {
    [all] => {
        $crate::SliceDim {
            start: 0,
            step: 1,
            len: udim::MAX,
        }
    };
    [$start:expr; $step:expr; $len:expr] => {
        $crate::SliceDim {
            start: $start,
            step: $step,
            len: $len,
        }
    };
}

fn build(meta: &[SliceDim], input: &[udim]) -> (Shape, Affine) {
    assert_eq!(input.len(), meta.len());
    assert!(meta
        .iter()
        .zip(input)
        .all(|(d, &i)| { (0..i).contains(&d.start) }));

    let shape = meta
        .iter()
        .zip(input)
        .map(|(d, &i)| {
            let distance = if d.step > 0 { i - d.start } else { d.start };
            let step = d.step.unsigned_abs();
            ((distance + step - 1) / step).min(d.len)
        })
        .collect::<Shape>();

    let n = meta.len();
    let affine = Affine::from_fn(n + 1, n + 1, |r, c| {
        if r == n {
            meta.get(c).map_or(1, |d| d.start as _)
        } else if r == c {
            meta.get(c).map(|d| d.step).unwrap()
        } else {
            0
        }
    });
    (shape, affine)
}

#[test]
fn test() {
    let (shape, affine) = build(&[slice![2;1;2], slice![0;1;4], slice![1;2;3]], &[5, 6, 7]);
    assert_eq!(shape.as_slice(), &[2, 4, 3]);
    assert_eq!(
        affine.as_slice(),
        &[
            // column major
            1, 0, 0, 2, //
            0, 1, 0, 0, //
            0, 0, 2, 1, //
            0, 0, 0, 1, //
        ]
    );
}
