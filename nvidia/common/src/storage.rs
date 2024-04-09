use cuda::{Context, ContextSpore, DevMem, DevMemSpore, Stream};
use std::sync::Arc;
use tensor::{udim, DataType, LocalSplitable, Tensor};

pub struct Cache {
    pub context: Arc<Context>,
    pub mem: DevMemSpore,
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.context.apply(|ctx| unsafe { self.mem.kill(ctx) });
    }
}

#[inline]
pub fn tensor<'ctx>(
    dt: DataType,
    shape: &[udim],
    stream: &Stream<'ctx>,
) -> Tensor<LocalSplitable<DevMem<'ctx>>> {
    Tensor::alloc(dt, shape, |l| stream.malloc::<u8>(l).into())
}
