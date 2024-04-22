mod broadcast;
mod compatibility;
mod data_type;
mod fmt;
mod pattern;
mod reshape;
mod slice;
mod split;
mod tensor;
mod transpose;

#[allow(non_camel_case_types)]
pub type udim = u32;

#[allow(non_camel_case_types)]
pub type idim = i32;

pub use compatibility::Compatibility;
pub use data_type::{DataType, Ty};
pub use nalgebra::DVector;
pub use pattern::{expand_indices, idx_strides, Affine, Shape};
pub use slice::SliceDim;
pub use split::{LocalSplitable, Splitable};
pub use tensor::Tensor;

use std::mem::{align_of, size_of, size_of_val};

pub fn reslice<T, U>(src: &[T]) -> &[U] {
    let ptr = src.as_ptr_range();
    let align = align_of::<U>();
    assert_eq!(ptr.start.align_offset(align), 0);
    assert_eq!(ptr.end.align_offset(align), 0);
    unsafe { std::slice::from_raw_parts(ptr.start.cast(), size_of_val(src) / size_of::<U>()) }
}

pub fn reslice_mut<T, U>(src: &mut [T]) -> &mut [U] {
    let ptr = src.as_mut_ptr_range();
    let align = align_of::<U>();
    assert_eq!(ptr.start.align_offset(align), 0);
    assert_eq!(ptr.end.align_offset(align), 0);
    unsafe { std::slice::from_raw_parts_mut(ptr.start.cast(), size_of_val(src) / size_of::<U>()) }
}
