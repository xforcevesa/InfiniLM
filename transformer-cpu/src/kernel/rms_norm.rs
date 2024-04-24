use gemm::f16;
use std::{
    iter::zip,
    ops::{Deref, DerefMut, Mul},
    slice::{from_raw_parts, from_raw_parts_mut},
};
use tensor::{DataType, Tensor};
use transformer::BetweenF32;

pub fn rms_norm<T, U, V>(o: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>, epsilon: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
    V: Deref<Target = [u8]>,
{
    let &[n, d] = o.shape() else { panic!() };
    let dt = o.data_type();

    assert_eq!(x.data_type(), dt);
    assert_eq!(w.data_type(), dt);
    assert_eq!(o.shape(), x.shape());
    assert_eq!(w.shape(), &[d]);
    assert!(o.contiguous_len() >= 1);
    assert!(x.contiguous_len() >= 1);
    assert!(w.is_contiguous());

    let ptr_o = o.locate_start_mut();
    let ptr_x = x.locate_start();
    let ptr_w = w.locate_start();

    let stride_o = o.strides()[0] as usize;
    let stride_x = x.strides()[0] as usize;
    let n = n as usize;
    let d = d as usize;

    match dt {
        DataType::F16 => rms_norm_op::<f16>(ptr_o, stride_o, ptr_x, stride_x, ptr_w, n, d, epsilon),
        DataType::F32 => rms_norm_op::<f32>(ptr_o, stride_o, ptr_x, stride_x, ptr_w, n, d, epsilon),
        _ => unreachable!("unsupported data type \"{dt:?}\""),
    }
}

fn rms_norm_op<T: Mul<Output = T> + BetweenF32 + Copy>(
    o: *mut u8,
    stride_o: usize,
    x: *const u8,
    stride_x: usize,
    w: *const u8,
    n: usize,
    d: usize,
    epsilon: f32,
) {
    let o = o.cast::<T>();
    let x = x.cast::<T>();
    let w = unsafe { from_raw_parts(w.cast::<T>(), d) };
    for i in 0..n {
        let o = unsafe { from_raw_parts_mut(o.add(stride_o * i), d) };
        let x = unsafe { from_raw_parts(x.add(stride_x * i), d) };

        // (Σx^2 / n + δ)^(-1/2)
        let mut len = 0usize;
        let mut sum = 0.0f32;
        for x in x.iter().map(T::get) {
            len += 1;
            sum += x * x;
        }
        let k = T::cast((sum / (len as f32) + epsilon).sqrt().recip());

        zip(o, zip(x, w)).for_each(|(o, (x, w))| *o = *w * (k * *x));
    }
}
