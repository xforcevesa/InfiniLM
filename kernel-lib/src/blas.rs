use std::{
    ffi::{c_int, c_longlong},
    mem::swap,
    os::raw::c_void,
};
use tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub batch: c_int,
    pub stride: c_longlong,
    pub r: c_int,
    pub c: c_int,
    pub rs: c_int,
    pub cs: c_int,
    pub base: *mut c_void,
}

impl Matrix {
    pub fn new<T>(tensor: &Tensor<T>, f: impl FnOnce(&T) -> *mut c_void) -> Self {
        let strides = tensor.strides();
        let base = (f(tensor.physical()) as usize + tensor.bytes_offset() as usize) as _;
        match tensor.shape() {
            &[r, c] => Self {
                batch: 1,
                stride: 0,
                r: r as _,
                c: c as _,
                rs: strides[0] as _,
                cs: strides[1] as _,
                base,
            },
            &[batch, r, c] => Self {
                batch: batch as _,
                stride: if batch == 1 { 0 } else { strides[0] as _ },
                r: r as _,
                c: c as _,
                rs: strides[1] as _,
                cs: strides[2] as _,
                base,
            },
            s => panic!("Invalid matrix shape: {s:?}"),
        }
    }
}

impl Matrix {
    #[inline]
    pub fn match_batch(&self, batch: c_int) -> bool {
        self.batch == batch || self.batch == 1
    }

    #[inline]
    pub fn transpose(&mut self) {
        swap(&mut self.r, &mut self.c);
        swap(&mut self.rs, &mut self.cs);
    }

    #[inline]
    pub fn ld(&self) -> c_int {
        if self.rs == 1 {
            self.cs
        } else if self.cs == 1 {
            self.rs
        } else {
            panic!("Matrix is not contiguous");
        }
    }
}
