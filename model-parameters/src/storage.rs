use std::{ops::Range, sync::Arc};

type Blob = dyn 'static + AsRef<[u8]>;

#[derive(Clone)]
pub struct Storage {
    data: Arc<Blob>,
    range: Range<usize>,
}

impl Storage {
    #[inline]
    pub fn new(data: Arc<Blob>, offset: usize, len: usize) -> Self {
        Self {
            data,
            range: offset..offset + len,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data.as_ref().as_ref()[self.range.clone()]
    }
}
