use memmap2::MmapMut;
use std::{
    ops::{Deref, Range},
    sync::Arc,
};
use tensor::{udim, DataType};

pub trait HostMem: Deref<Target = [u8]> + 'static + Send + Sync {}
impl<T> HostMem for T where T: Deref<Target = [u8]> + 'static + Send + Sync {}

#[derive(Clone)]
pub struct Storage {
    pub(crate) data: Arc<dyn HostMem>,
    pub(crate) range: Range<usize>,
}

impl Deref for Storage {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data[self.range.clone()]
    }
}

#[inline]
pub fn map_anon(data_type: DataType, shape: &[udim]) -> MmapMut {
    let size = shape.iter().product::<udim>() as usize * data_type.size();
    MmapMut::map_anon(size).unwrap()
}
