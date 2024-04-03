#![cfg(detected_nccl)]

#[test]
fn test_load() {
    use cuda::{ContextResource, ContextSpore, Device};
    use std::{io::ErrorKind::NotFound, time::Instant};
    use transformer::{Distributer, Llama2, Memory, SafeTensorError};

    const N: usize = 1;

    cuda::init();
    if Device::count() < N {
        return;
    }
    let devices = (0..N as _).map(Device::new).collect::<Vec<_>>();
    let contexts = devices
        .iter()
        .map(Device::retain_primary)
        .collect::<Vec<_>>();
    let align = devices.iter().map(Device::alignment).max().unwrap();

    let time = Instant::now();
    let safetensors = Memory::load_safetensors_from_dir("../../../TinyLlama-1.1B-Chat-v1.0");
    println!("mmap {:?}", time.elapsed());

    let model = match safetensors {
        Ok(m) => m,
        Err(SafeTensorError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let nlayers = model.num_hidden_layers();
    let mut matrix = Vec::with_capacity(contexts.len() * nlayers);

    let distributer = Distributer::new(&model, contexts.len(), align);
    let time = Instant::now();
    for (i, context) in contexts.iter().enumerate() {
        context.apply(|ctx| {
            let stream = ctx.stream();
            for layer in 0..nlayers {
                matrix.push(
                    stream
                        .from_host(distributer.distribute(layer, i).as_slice())
                        .sporulate(),
                );
            }
        });
    }
    println!("distribute {:?}", time.elapsed());

    let time = Instant::now();
    for (i, context) in contexts.iter().enumerate() {
        context.apply(|ctx| {
            for element in &mut matrix[i * nlayers..][..nlayers] {
                unsafe { element.kill(ctx) };
            }
        });
    }
    println!("kill {:?}", time.elapsed());
}
