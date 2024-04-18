use common::safe_tensors::SharedTensor;
use std::{ops::Deref, sync::Arc};

pub trait HostMem: Deref<Target = [u8]> + 'static + Send + Sync {}
impl<T> HostMem for T where T: Deref<Target = [u8]> + 'static + Send + Sync {}

#[derive(Clone)]
pub enum Storage {
    SafeTensor(SharedTensor),
    Others(Arc<dyn HostMem>),
}

impl Deref for Storage {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Storage::SafeTensor(tensor) => tensor,
            Storage::Others(blob) => blob,
        }
    }
}
