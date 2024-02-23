use crate::{expand_indices, idim, idx_strides, udim, DataType, Tensor};
use half::{bf16, f16};
use std::{fmt, ops::Deref};

pub trait DataFmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl DataFmt for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == &f16::ZERO {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.to_f32())
        }
    }
}

impl DataFmt for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == &bf16::ZERO {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.to_f32())
        }
    }
}

impl DataFmt for f32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == &0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self)
        }
    }
}

impl DataFmt for f64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == &0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self)
        }
    }
}

impl<Physical: Deref<Target = [u8]>> fmt::Display for Tensor<Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        macro_rules! write_tensor {
            ($ty:ty) => {
                write_tensor(
                    f,
                    self.locate_start().cast::<$ty>(),
                    self.shape(),
                    self.strides(),
                )
            };
        }
        match self.data_type() {
            DataType::F16 => write_tensor!(f16),
            DataType::BF16 => write_tensor!(bf16),
            DataType::F32 => write_tensor!(f32),
            DataType::F64 => write_tensor!(f64),
            _ => todo!(),
        }
    }
}

fn write_tensor<T: DataFmt>(
    f: &mut fmt::Formatter<'_>,
    ptr: *const T,
    shape: &[udim],
    strides: &[idim],
) -> fmt::Result {
    assert_eq!(shape.len(), strides.len());
    match shape {
        [] => {
            writeln!(f, "<>")?;
            write_matrix(f, ptr, (1, 1), (1, 1))
        }
        [len] => {
            writeln!(f, "<{len}>")?;
            write_matrix(f, ptr, (*len, 1), (strides[0], 1))
        }
        [rows, cols] => {
            writeln!(f, "<{rows}x{cols}>")?;
            write_matrix(f, ptr, (*rows, *cols), (strides[0], strides[1]))
        }
        [batch @ .., rows, cols] => {
            let (strides, tail) = strides.split_at(batch.len());
            let rs = tail[0];
            let cs = tail[1];
            let (n, idx_strides) = idx_strides(batch);
            for i in 0..n {
                let indices = expand_indices(i, &idx_strides, &[]);
                writeln!(
                    f,
                    "<{rows}x{cols}>[{}]",
                    indices
                        .iter()
                        .map(idim::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                )?;
                let ptr = unsafe {
                    ptr.offset(
                        indices
                            .iter()
                            .zip(strides)
                            .map(|(&a, &b)| a as isize * b as isize)
                            .sum(),
                    )
                };
                write_matrix(f, ptr, (*rows, *cols), (rs, cs))?;
            }
            Ok(())
        }
    }
}

fn write_matrix<T: DataFmt>(
    f: &mut fmt::Formatter<'_>,
    ptr: *const T,
    shape: (udim, udim),
    strides: (idim, idim),
) -> fmt::Result {
    let rows = shape.0 as usize;
    let cols = shape.1 as usize;
    let rs = strides.0 as usize;
    let cs = strides.1 as usize;
    for r in 0..rows {
        for c in 0..cols {
            unsafe { &*ptr.add(r * rs + c * cs) }.fmt(f)?;
            write!(f, " ")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

#[test]
fn test_fmt() {
    use crate::{reslice, slice, DataType, Tensor};

    let shape = [2, 3, 4];
    let data = Vec::from_iter((0..24).map(|x| x as f32));
    let data = reslice(&data);

    let t = Tensor::new(DataType::F32, &shape, data);
    println!("{t}");

    let t = t.reshape(&[2, 3, 2, 2]);
    println!("{t}");

    let t = t.reshape(&[6, 4]);
    println!("{t}");

    let t = t.transpose(&[1, 0]);
    println!("{t}");

    let t = t.slice(&[slice![0; 2; 2], slice![1; 2; 3]]);
    println!("{t}");

    let t = t.transpose(&[1, 0]);
    println!("{t}");

    let t = t.reshape(&[3, 1, 2]);
    println!("{t}");

    let t = t.broadcast(&[3, 3, 2]);
    println!("{t}");

    let mut t_ = Tensor::new(t.data_type(), t.shape(), vec![0u8; t.bytes_size()]);
    t.reform_to(&mut t_);
    println!("{t_}");
}
