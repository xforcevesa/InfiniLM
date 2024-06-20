use common_nv::{
    cuda::{Context, ContextResource, ContextSpore, DevMemSpore, Device, Stream, StreamSpore},
    DropOption,
};
use std::sync::Arc;

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
        self.context.apply(|ctx| drop(self.compute.sprout(ctx)));
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
            self.mem.sprout(stream.ctx()).drop_on(stream);
        });
    }
}
