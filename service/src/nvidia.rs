use std::{fs::File, path::Path, time::Instant};
use transformer_nv::{cuda, NvidiaTransformer};

pub fn transformer(model_dir: impl AsRef<Path>, device: i32) -> NvidiaTransformer {
    cuda::init();

    let time = Instant::now();
    let model_dir = model_dir.as_ref();
    let config = File::open(model_dir.join("config.json")).unwrap();
    let safetensors = File::open(model_dir.join("model.safetensors")).unwrap();
    info!("open file {:?}", time.elapsed());

    let time = Instant::now();
    let dev = cuda::Device::new(device);
    dev.set_mempool_threshold(u64::MAX);
    let transformer = NvidiaTransformer::new(config, safetensors, usize::MAX, dev);
    info!("build transformer ... {:?}", time.elapsed());

    transformer
}
