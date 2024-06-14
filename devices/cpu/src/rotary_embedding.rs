use common_devices::layout;
use operators::{
    common_cpu::ThisThread,
    rope::{
        common_cpu::{Operator as Rope, Scheme as RopeScheme},
        LayoutAttrs,
    },
    Operator, Scheme, F16,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

/// - t:   `[num_token, num_head, head_dim]`
/// - pos: `[num_token]`
pub fn rotary_embedding<T, U>(t: &mut Tensor<T>, pos: &Tensor<U>, theta: f32)
where
    T: DerefMut<Target = [u8]>,
    U: Deref<Target = [u8]>,
{
    RopeScheme::new(
        &Rope::new(&F16).unwrap(),
        LayoutAttrs {
            t: layout(t),
            pos: layout(pos),
        },
    )
    .unwrap()
    .launch(
        &(
            t.physical_mut().as_mut_ptr(),
            pos.physical().as_ptr(),
            theta,
        ),
        &ThisThread,
    );
}
