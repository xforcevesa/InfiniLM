use crate::Tensor;
use std::iter::zip;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Compatibility {
    Same,
    Squeeze,
    Reform,
    None,
}

impl Compatibility {
    pub fn between<T, U>(a: &Tensor<T>, b: &Tensor<U>) -> Self {
        if a.data_type != b.data_type {
            return Self::None;
        }

        let mut actual_a = zip(&a.shape, a.pattern.0.as_slice()).filter(|(&d, _)| d > 1);
        let mut actual_b = zip(&b.shape, b.pattern.0.as_slice()).filter(|(&d, _)| d > 1);
        let mut squeeze = true;
        loop {
            match (actual_a.next(), actual_b.next()) {
                (Some((da, sa)), Some((db, sb))) => {
                    if da != db {
                        return Self::None;
                    }
                    if sa != sb {
                        squeeze = false;
                    }
                }
                (Some(_), None) | (None, Some(_)) => return Self::None,
                (None, None) => break,
            }
        }
        if squeeze {
            if a.shape == b.shape {
                Self::Same
            } else {
                Self::Squeeze
            }
        } else {
            Self::Reform
        }
    }
}
