use common_nv::cuda::{Context, ContextResource, DevMemSpore, Device, Stream, StreamSpore};
use std::sync::Arc;

pub(super) struct Resource {
    context: Context,
    compute: StreamSpore,
}

impl Resource {
    #[inline]
    pub fn new(device: &Device) -> Self {
        let context = device.retain_primary();
        let compute = context.apply(|ctx| ctx.stream().sporulate());
        Self { context, compute }
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&Stream) -> T) -> T {
        self.context
            .apply(|ctx| f(&unsafe { ctx.sprout(&self.compute) }))
    }
}

impl Drop for Resource {
    #[inline]
    fn drop(&mut self) {
        self.context
            .apply(|ctx| unsafe { ctx.kill(&mut self.compute) });
    }
}

pub struct Cache {
    res: Arc<Resource>,
    pub(super) mem: DevMemSpore,
}

impl Cache {
    #[inline]
    pub(super) fn new(res: &Arc<Resource>, len: usize) -> Self {
        Self {
            res: res.clone(),
            mem: res.apply(|compute| compute.malloc::<u8>(len).sporulate()),
        }
    }
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.res.context.apply(|ctx| unsafe {
            self.mem.kill_on(&ctx.sprout(&self.res.compute));
        });
    }
}
