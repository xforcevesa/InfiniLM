use std::{fs::File, path::Path, sync::Arc, time::Instant};
use transformer_nvidia::{cuda, NvidiaTransformer};

pub fn transformer(model_dir: impl AsRef<Path>, device: i32) -> NvidiaTransformer {
    cuda::init();
    let device = cuda::Device::new(device);
    device.set_mempool_threshold(u64::MAX);
    let model_dir = model_dir.as_ref();

    let time = Instant::now();
    let config = File::open(model_dir.join("config.json")).unwrap();
    let safetensors = File::open(model_dir.join("model.safetensors")).unwrap();
    info!("open file {:?}", time.elapsed());

    let time = Instant::now();
    let context = Arc::new(device.context());
    let transformer = NvidiaTransformer::new(config, safetensors, usize::MAX, context.clone());
    info!("build transformer ... {:?}", time.elapsed());

    transformer
}
