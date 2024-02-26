#![cfg(detected_cuda)]

use cuda::{driver, Context};
use std::{
    ptr::{null_mut, NonNull},
    sync::Arc,
};

pub extern crate model_parameters;

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
    use model_parameters::{Memory, SafeTensorError};
    use std::{io::ErrorKind::NotFound, time::Instant};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };

    let t0 = Instant::now();
    let safetensors = Memory::load_safetensors("../../TinyLlama-1.1B-Chat-v1.0_F16");
    let t1 = Instant::now();
    println!("mmap {:?}", t1 - t0);

    let safetensors = match safetensors {
        Ok(m) => m,
        Err(SafeTensorError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    dev.context().apply(|ctx| {
        let t0 = Instant::now();
        let _model = Memory::realloc_with(&safetensors, HostAllocator(ctx.clone_ctx()));
        let t1 = Instant::now();
        println!("realloc {:?}", t1 - t0);
    });
}
