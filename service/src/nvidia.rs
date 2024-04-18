use std::{path::Path, time::Instant};

pub fn transformer(model_dir: impl AsRef<Path>, device: i32) -> transformer_nv::Transformer {
    use transformer_nv::{cuda, Transformer};

    let time = Instant::now();
    cuda::init();
    let dev = cuda::Device::new(device);
    dev.set_mempool_threshold(u64::MAX);
    let transformer = Transformer::new(model_dir, usize::MAX, dev);
    info!("build transformer ... {:?}", time.elapsed());

    transformer
}

#[cfg(detected_nccl)]
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
