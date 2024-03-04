use gemm::f16;
use std::ops::{Deref, DerefMut};
use tensor::{idim, DVector, Tensor};

pub fn swiglu<T, U>(mut gate: Tensor<T>, up: &Tensor<U>)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let &[seq_len, di] = gate.shape() else {
        panic!("gate shape: {:?}", gate.shape());
    };
    assert_eq!(gate.data_type(), up.data_type());
    assert_eq!(up.shape(), &[seq_len, di]);
    assert!(gate.contiguous_len() >= 1);
    assert!(up.contiguous_len() >= 1);

    for i in 0..seq_len {
        let indices = DVector::from_vec(vec![i as idim, 0, 1]);
        let gate = gate.locate_mut(&indices.as_view()).unwrap();
        let gate = unsafe { std::slice::from_raw_parts_mut(gate.cast::<f16>(), di as usize) };
        let up = up.locate(&indices.as_view()).unwrap();
        let up = unsafe { std::slice::from_raw_parts(up.cast::<f16>(), di as usize) };
        for (gate, up) in gate.iter_mut().zip(up) {
            let x = gate.to_f32();
            let y = up.to_f32();

            #[inline(always)]
            fn sigmoid(x: f32) -> f32 {
                1. / (1. + (-x).exp())
            }

            *gate = f16::from_f32(x * sigmoid(x) * y);
        }
    }
}
