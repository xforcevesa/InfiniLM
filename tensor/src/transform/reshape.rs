use crate::{idim, idx_strides, pattern::Pattern, udim, Shape, Tensor};
use nalgebra::DVector;
use std::iter::zip;

impl<Physical> Tensor<Physical> {
    pub fn reshape(self, shape: &[udim]) -> Self {
        assert_eq!(self.size() as udim, shape.iter().product::<udim>());
        if self.is_contiguous() {
            // reshape: 张量物理连续，直接修改形状和模式
            assert_eq!(
                self.shape.iter().product::<udim>(),
                shape.iter().product::<udim>(),
            );
            return Self {
                data_type: self.data_type,
                shape: Shape::from_slice(shape),
                pattern: Pattern::from_shape(shape, self.pattern.offset()),
                physical: self.physical,
            };
        }

        fn remove1(shape: &[udim]) -> Shape {
            shape.iter().filter(|&&d| d > 1).copied().collect::<Shape>()
        }

        let current = remove1(&self.shape);
        let target = remove1(shape);
        let same_head = zip(&current, &target).take_while(|(a, b)| a == b).count();
        if same_head == current.len() {
            // squeeze: 张量形状只增减一些 1，扫描，将新增的 1 模式设置 0
            let mut i = 0;
            let mut pattern = Vec::with_capacity(shape.len() + 1);
            for &d in shape {
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match self.shape[i] {
                            1 => i += 1,
                            _ => break self.pattern.0[i],
                        }
                    });
                    debug_assert_eq!(self.shape[i], d);
                    i += 1;
                }
            }
            pattern.push(self.pattern.offset());
            return Self {
                data_type: self.data_type,
                shape: Shape::from_slice(shape),
                pattern: Pattern(DVector::from_vec(pattern)),
                physical: self.physical,
            };
        }

        let same_tail = zip(
            current[same_head..].iter().rev(),
            target[same_head..].iter().rev(),
        )
        .take_while(|(a, b)| a == b)
        .count();
        if same_head + same_tail + 1 == current.len() {
            // split: 原本的一个维度拆成多个，支持拆分物理连续的那一个维度
            let axis = same_head;
            let insert_dims = &target[axis..target.len() - same_tail];

            let mut i = 0;
            let mut j = 0;
            let mut k = 0;
            let mut pattern = Vec::with_capacity(shape.len() + 1);
            while j < same_head {
                let d = shape[k];
                k += 1;
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match self.shape[i] {
                            1 => i += 1,
                            _ => break self.pattern.0[i],
                        }
                    });
                    debug_assert_eq!(self.shape[i], d);
                    debug_assert_eq!(current[j], d);
                    debug_assert_eq!(target[j], d);
                    i += 1;
                    j += 1;
                }
            }

            while self.shape[i] == 1 {
                i += 1;
            }
            assert_eq!(self.pattern.0[i], 1);
            i += 1;

            let (_, insert_pattern) = idx_strides(insert_dims);
            let mut l = 0;
            while j < same_head + insert_dims.len() {
                let d = shape[k];
                k += 1;
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match insert_dims[l] {
                            1 => l += 1,
                            _ => break insert_pattern[l] as idim,
                        }
                    });
                    debug_assert_eq!(insert_dims[l], d);
                    debug_assert_eq!(target[j], d);
                    l += 1;
                    j += 1;
                }
            }

            while k < shape.len() {
                let d = shape[k];
                k += 1;
                if d == 1 {
                    pattern.push(0);
                } else {
                    pattern.push(loop {
                        match self.shape[i] {
                            1 => i += 1,
                            _ => break self.pattern.0[i],
                        }
                    });
                    debug_assert_eq!(self.shape[i], d);
                    debug_assert_eq!(current[j], d);
                    debug_assert_eq!(target[j], d);
                    i += 1;
                    j += 1;
                }
            }

            pattern.push(self.pattern.offset());
            return Self {
                data_type: self.data_type,
                shape: Shape::from_slice(shape),
                pattern: Pattern(DVector::from_vec(pattern)),
                physical: self.physical,
            };
        }
        panic!("unsupported reshape");
    }
}
