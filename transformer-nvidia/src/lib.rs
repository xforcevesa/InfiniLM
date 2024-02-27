#![cfg(detected_cuda)]
#![allow(dead_code)]

mod parameters;

use cuda::{driver, Context, Stream};
use model_parameters::Memory;
use parameters::{LayersParameters, ModelParameters};
use std::{
    ptr::{null_mut, NonNull},
    sync::Arc,
};

pub extern crate model_parameters;

pub struct Transformer<'a> {
    host: &'a Memory,
    model: ModelParameters,
    layers: LayersParameters,
}

impl<'a> Transformer<'a> {
    pub fn new(host: &'a Memory, stream: &Stream) -> Self {
        Self {
            host,
            model: ModelParameters::new(host, stream),
            layers: LayersParameters::new(3, host, stream),
        }
    }
}

struct HostAllocator(Arc<Context>);

impl model_parameters::Allocator for HostAllocator {
    #[inline]
    unsafe fn allocate(&self, size: usize) -> NonNull<u8> {
        let mut ptr = null_mut();
        self.0.apply(|_| driver!(cuMemHostAlloc(&mut ptr, size, 0)));
        NonNull::new(ptr.cast()).unwrap()
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>) {
        self.0
            .apply(|_| driver!(cuMemFreeHost(ptr.as_ptr().cast())));
    }
}

#[test]
fn test_load() {
    use model_parameters::Memory;
    use std::{
        fs::File,
        io::{ErrorKind::NotFound, Read},
        path::Path,
        time::Instant,
    };

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };

    let model_dir = Path::new("../../TinyLlama-1.1B-Chat-v1.0_F16");
    let time = Instant::now();

    let config = File::open(model_dir.join("config.json"));
    let config = match config {
        Ok(f) => f,
        Err(e) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let safetensors = File::open(model_dir.join("model.safetensors"));
    let mut safetensors = match safetensors {
        Ok(f) => f,
        Err(e) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };
    println!("open file {:?}", time.elapsed());

    dev.set_mempool_threshold(u64::MAX);
    dev.context().apply(|ctx| {
        let time = Instant::now();
        let host = {
            let len = safetensors.metadata().unwrap().len() as usize;
            let mut host_ptr = null_mut();
            driver!(cuMemHostAlloc(&mut host_ptr, len, 0));
            unsafe { std::slice::from_raw_parts_mut(host_ptr.cast::<u8>(), len) }
        };
        safetensors.read_exact(host).unwrap();
        drop(safetensors);
        println!("read to host {:?}", time.elapsed());

        let time = Instant::now();
        let host = Memory::load_safetensors(config, host, false).unwrap();
        println!("load {:?}", time.elapsed());

        let stream = ctx.stream();

        let time = Instant::now();
        let _transformer = Transformer::new(&host, &stream);
        println!("build model host: {:?}", time.elapsed());
    });
}
