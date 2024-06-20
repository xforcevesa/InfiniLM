use common_cn::{
    cndrv::{Context, ContextResource, DevMemSpore},
    DropOption,
};
use std::sync::Arc;

pub struct Cache {
    res: Arc<Context>,
    pub(super) mem: DropOption<DevMemSpore>,
}

impl Cache {
    #[inline]
    pub(super) fn new(res: &Arc<Context>, len: usize) -> Self {
        Self {
            res: res.clone(),
            mem: res.apply(|ctx| ctx.malloc::<u8>(len).sporulate()).into(),
        }
    }
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.res.apply(|ctx| drop(self.mem.sprout(ctx)));
    }
}
