﻿use crate::{idim, udim, DataType, Tensor};
use half::{bf16, f16};
use std::fmt;

fn write_tensor<T: fmt::LowerExp>(
    to: &mut fmt::Formatter<'_>,
    ptr: *const T,
    shape: &[udim],
    strides: &[idim],
) -> fmt::Result {
    assert_eq!(shape.len(), strides.len());
    match shape {
        [] => {
            writeln!(to, "<>")?;
            write_matrix(to, ptr, (1, 1), (1, 1))
        }
        [len] => {
            writeln!(to, "<{len}>")?;
            write_matrix(to, ptr, (*len, 1), (strides[0], 1))
        }
        [rows, cols] => {
            writeln!(to, "<{rows}x{cols}>")?;
            write_matrix(to, ptr, (*rows, *cols), (strides[0], strides[1]))
        }
        [batch @ .., rows, cols] => {
            let (strides, tail) = strides.split_at(batch.len());
            let rs = tail[0];
            let cs = tail[1];
            let mut idx_strides = vec![0; batch.len()];
            idx_strides[batch.len() - 1] = 1;
            for i in (1..batch.len()).rev() {
                idx_strides[i - 1] = idx_strides[i] * shape[i];
            }
            for i in 0..batch[0] * idx_strides[0] as udim {
                let mut which = vec![0; strides.len()];
                let mut rem = i;
                for (j, &s) in idx_strides.iter().enumerate() {
                    which[j] = rem / s;
                    rem %= s;
                }
                writeln!(
                    to,
                    "<{rows}x{cols}>[{}]",
                    which
                        .iter()
                        .map(udim::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                )?;
                let ptr = unsafe {
                    ptr.offset(
                        which
                            .iter()
                            .zip(strides)
                            .map(|(&a, &b)| a as isize * b as isize)
                            .sum(),
                    )
                };
                write_matrix(to, ptr, (*rows, *cols), (rs, cs))?;
            }
            Ok(())
        }
    }
}

fn write_matrix<T: fmt::LowerExp>(
    to: &mut fmt::Formatter<'_>,
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
            write!(to, "{:<8.3e} ", unsafe { &*ptr.add(r * rs + c * cs) })?;
        }
        writeln!(to)?;
    }
    Ok(())
}

impl<Physical: AsRef<[u8]>> fmt::Display for Tensor<Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let offset = self.offset() as usize;
        macro_rules! write_tensor {
            ($ty:ty) => {
                write_tensor(
                    f,
                    unsafe { self.as_slice().as_ptr().cast::<$ty>().add(offset) },
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

#[test]
fn test_fmt() {
    use crate::Shape;
    use crate::{Broadcast, DataType, Slice, SliceDim, Squeeze, SqueezeOp, Tensor, Transpose};
    use std::mem::size_of;

    let shape = [2, 3, 4];
    let data = [
        0.0f32, 1., 2., 3., //
        04., 05., 06., 07., //
        08., 09., 10., 11., //
        //
        12., 13., 14., 15., //
        16., 17., 18., 19., //
        20., 21., 22., 23., //
    ];
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * size_of::<f32>())
    };

    let t = Tensor::new(DataType::F32, &shape, data);
    println!("{t}");

    let t = t.reshape(Shape::from_slice(&[2, 3, 2, 2]));
    println!("{t}");

    let t = t.reshape(Shape::from_slice(&[6, 4]));
    println!("{t}");

    let t = t.apply(&Transpose::new(&[1, 0])).pop().unwrap();
    println!("{t}");

    let t = t
        .apply(&Slice::new(&[
            SliceDim {
                start: 0,
                step: 2,
                len: 2,
            },
            SliceDim {
                start: 1,
                step: 2,
                len: 3,
            },
        ]))
        .pop()
        .unwrap();
    println!("{t}");

    let t = t.apply(&Transpose::new(&[1, 0])).pop().unwrap();
    println!("{t}");

    let t = t
        .apply(&Squeeze::new(&[
            SqueezeOp::Skip,
            SqueezeOp::Insert,
            SqueezeOp::Skip,
        ]))
        .pop()
        .unwrap();
    println!("{t}");

    let t = t.apply(&Broadcast::new(&[3, 3, 2])).pop().unwrap();
    println!("{t}");
}