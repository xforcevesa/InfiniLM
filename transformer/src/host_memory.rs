use std::{
    ops::{Deref, Range},
    sync::Arc,
};

#[derive(Clone)]
pub struct HostMemory<'a> {
    data: Arc<dyn Deref<Target = [u8]> + 'a>,
    pub(crate) range: Range<usize>,
}

impl Deref for HostMemory<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data[self.range.clone()]
    }
}

impl<'a> HostMemory<'a> {
    #[inline]
    pub const fn new(data: Arc<dyn Deref<Target = [u8]> + 'a>, offset: usize, len: usize) -> Self {
        Self {
            data,
            range: offset..offset + len,
        }
    }

    #[inline]
    pub fn from_blob<T>(data: T) -> Self
    where
        T: Deref<Target = [u8]> + 'a,
    {
        let len = data.as_ref().len();
        Self {
            data: Arc::new(data),
            range: 0..len,
        }
    }
}
