use std::{
    ops::{Deref, Range},
    sync::Arc,
};

#[derive(Clone)]
pub struct HostMemory {
    data: Arc<dyn Deref<Target = [u8]>>,
    pub(crate) range: Range<usize>,
}

impl Deref for HostMemory {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data[self.range.clone()]
    }
}

impl HostMemory {
    #[inline]
    pub const fn new(data: Arc<dyn Deref<Target = [u8]>>, offset: usize, len: usize) -> Self {
        Self {
            data,
            range: offset..offset + len,
        }
    }

    #[inline]
    pub fn from_blob(data: impl 'static + Deref<Target = [u8]>) -> Self {
        let len = data.as_ref().len();
        Self {
            data: Arc::new(data),
            range: 0..len,
        }
    }
}
