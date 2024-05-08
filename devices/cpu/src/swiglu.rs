use common::{f16, BetweenF32};
use std::ops::{Deref, DerefMut};
use tensor::{idim, DVector, DataType, Tensor};

pub fn swiglu<T, U>(gate: &mut Tensor<T>, up: &Tensor<U>)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    let dt = gate.data_type();
    assert_eq!(up.data_type(), dt);

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
        let up = up.locate(&indices.as_view()).unwrap();

        match dt {
            DataType::F16 => typed::<f16>(gate, up, di as _),
            DataType::F32 => typed::<f32>(gate, up, di as _),
            _ => unreachable!(),
        }
    }
}

fn typed<T>(gate: *mut u8, up: *const u8, di: usize)
where
    T: BetweenF32,
{
    let gate = unsafe { std::slice::from_raw_parts_mut(gate.cast::<T>(), di) };
    let up = unsafe { std::slice::from_raw_parts(up.cast::<T>(), di) };
    for (gate, up) in gate.iter_mut().zip(up) {
        let x = gate.get();
        let y = up.get();

        #[inline(always)]
        fn sigmoid(x: f32) -> f32 {
            1. / (1. + (-x).exp())
        }

        *gate = T::cast(x * sigmoid(x) * y);
    }
}
