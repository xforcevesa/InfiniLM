use common_nv::cuda::{
    Context, ContextResource, ContextSpore, DevMemSpore, Device, Stream, StreamSpore,
};
use std::sync::Arc;

pub(super) struct DropOption<T>(Option<T>);

impl<T> From<T> for DropOption<T> {
    fn from(value: T) -> Self {
        Self(Some(value))
    }
}

impl<T> DropOption<T> {
    pub fn as_ref(&self) -> &T {
        self.0.as_ref().unwrap()
    }

    pub fn as_mut(&mut self) -> &mut T {
        self.0.as_mut().unwrap()
    }

    pub fn take(&mut self) -> T {
        self.0.take().unwrap()
    }
}

pub(super) struct Resource {
    context: Context,
    compute: DropOption<StreamSpore>,
}

impl Resource {
    #[inline]
    pub fn new(device: &Device) -> Self {
        let context = device.retain_primary();
        let compute = context.apply(|ctx| ctx.stream().sporulate());
        Self {
            context,
            compute: compute.into(),
        }
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&Stream) -> T) -> T {
        self.context
            .apply(|ctx| f(self.compute.as_ref().sprout_ref(ctx)))
    }
}

impl Drop for Resource {
    #[inline]
    fn drop(&mut self) {
        self.context
            .apply(|ctx| drop(self.compute.take().sprout(ctx)));
    }
}

pub struct Cache {
    res: Arc<Resource>,
    pub(super) mem: DropOption<DevMemSpore>,
}

impl Cache {
    #[inline]
    pub(super) fn new(res: &Arc<Resource>, len: usize) -> Self {
        Self {
            res: res.clone(),
            mem: res
                .apply(|compute| compute.malloc::<u8>(len).sporulate())
                .into(),
        }
    }
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.res.apply(|stream| {
            self.mem.take().sprout(stream.ctx()).drop_on(stream);
        });
    }
}
