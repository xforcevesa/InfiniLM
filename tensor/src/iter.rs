use crate::udim;

pub(crate) struct IndicesIterator {
    i: udim,
    end: udim,
    idx_strides: Vec<udim>,
}

impl IndicesIterator {
    pub fn new(shape: &[udim]) -> Self {
        let mut idx_strides = vec![0; shape.len()];
        idx_strides[shape.len() - 1] = 1;
        for i in (1..shape.len()).rev() {
            idx_strides[i - 1] = idx_strides[i] * shape[i];
        }
        Self {
            i: 0,
            end: shape[0] * idx_strides[0],
            idx_strides,
        }
    }
}

impl Iterator for IndicesIterator {
    type Item = Vec<udim>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.end {
            None
        } else {
            let ans = self
                .idx_strides
                .iter()
                .scan(self.i, |rem, &s| {
                    let q = *rem / s;
                    *rem %= s;
                    Some(q)
                })
                .collect();
            self.i += 1;
            Some(ans)
        }
    }
}
