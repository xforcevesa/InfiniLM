use std::{fs::File, path::Path, time::Instant};

pub fn transformer(model_dir: impl AsRef<Path>, device: i32) -> transformer_nv::Transformer {
    use transformer_nv::{cuda, Transformer};
    cuda::init();

    let time = Instant::now();
    let model_dir = model_dir.as_ref();
    let config = File::open(model_dir.join("config.json")).unwrap();
    let safetensors = File::open(model_dir.join("model.safetensors")).unwrap();
    info!("open file {:?}", time.elapsed());

    let time = Instant::now();
    let dev = cuda::Device::new(device);
    dev.set_mempool_threshold(u64::MAX);
    let transformer = Transformer::new(config, safetensors, usize::MAX, dev);
    info!("build transformer ... {:?}", time.elapsed());

    transformer
}

pub fn distributed(
    model_dir: impl AsRef<Path>,
    devices: impl IntoIterator<Item = i32>,
) -> distributed::Transformer {
    use distributed::{cuda, Transformer};
    cuda::init();

    let time = Instant::now();
    let dev = devices
        .into_iter()
        .map(cuda::Device::new)
        .collect::<Vec<_>>();
    let transformer = Transformer::new(model_dir, &dev);
    info!("load {:?}", time.elapsed());

    transformer
}
