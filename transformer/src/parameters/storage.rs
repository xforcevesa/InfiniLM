use std::{
    ops::{Deref, Range},
    sync::Arc,
};

pub trait HostMem: Deref<Target = [u8]> + 'static + Send + Sync {}
impl<T> HostMem for T where T: Deref<Target = [u8]> + 'static + Send + Sync {}

#[derive(Clone)]
pub struct Storage {
    pub(crate) data: Arc<dyn HostMem>,
    pub(crate) range: Range<usize>,
}

impl Storage {
    #[inline]
    pub fn new(data: impl HostMem) -> Self {
        Self {
            range: 0..data.len(),
            data: Arc::new(data),
        }
    }
}

impl Deref for Storage {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data[self.range.clone()]
    }
}
