use std::{path::Path, time::Instant};
use transformer::{Memory, Transformer};
use transformer_cpu::CpuTransformer;

pub fn transformer(model_dir: impl AsRef<Path>) -> impl Transformer {
    let time = Instant::now();
    let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
    info!("load model ... {:?}", time.elapsed());

    let time = Instant::now();
    let transformer = CpuTransformer::new(model);
    info!("build transformer ... {:?}", time.elapsed());

    transformer
}
