use cublas::cublas;
use tensor::Tensor;

use crate::storage::DevMem;

pub fn mat_mul(
    handle: cublas::bindings::cublasHandle_t,
    c: &Tensor<DevMem>,
    beta: f32,
    a: &Tensor<DevMem>,
    b: &Tensor<DevMem>,
    alpha: f32,
) {
}
