#![cfg(detected_neuware)]

pub extern crate cndrv;

use cndrv::{ContextGuard, ContextSpore};

pub use tensor::Tensor;

pub struct DropOption<T>(Option<T>);

impl<T> From<T> for DropOption<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(Some(value))
    }
}

impl<T: ContextSpore> DropOption<T> {
    #[inline]
    pub fn as_ref(&self) -> &T {
        self.0.as_ref().unwrap()
    }

    #[inline]
    pub fn as_mut(&mut self) -> &mut T {
        self.0.as_mut().unwrap()
    }

    #[inline]
    pub fn sprout<'ctx>(&mut self, ctx: &'ctx ContextGuard) -> <T as ContextSpore>::Resource<'ctx> {
        self.0.take().unwrap().sprout(ctx)
    }
}

pub fn synchronize() {
    cndrv::init();
    for i in 0..cndrv::Device::count() {
        cndrv::Device::new(i as _)
            .acquire_shared()
            .apply(|ctx| ctx.synchronize());
    }
}
