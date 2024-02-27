#![cfg(detected_cuda)]

mod parameters;
mod storage;

use common::{upos, utok};
use cuda::Stream;
use model_parameters::Llama2;
use parameters::{LayersParameters, ModelParameters};
use storage::DevMem;
use tensor::{reslice, slice, udim, DataType, Tensor};

pub use storage::PageLockedMemory;

pub extern crate cuda;
pub extern crate model_parameters;

pub struct Transformer<'a> {
    host: &'a dyn Llama2,
    model: ModelParameters,
    layers: LayersParameters,
}

impl<'a> Transformer<'a> {
    pub fn new(host: &'a dyn Llama2, stream: &Stream) -> Self {
        Self {
            host,
            model: ModelParameters::new(host, stream),
            layers: LayersParameters::new(3, host, stream),
        }
    }

    #[allow(unused)]
    pub fn update(
        &self,
        tokens: &[utok],
        /*cache: &mut [LayerCache],*/ pos: upos,
        transfer: &Stream,
    ) {
        let seq_len = tokens.len() as udim;
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let di = self.host.intermediate_size() as udim;
        let dt = self.host.data_type();
        let epsilon = self.host.rms_norm_eps();
        let theta = self.host.rope_theta();
        let att_len = pos + seq_len;
        let cat_slice = &[slice![all], slice![pos; 1; seq_len], slice![all]];
        let att_slice = &[slice![all], slice![  0; 1; att_len], slice![all]];
        let pos = (pos..pos + seq_len).collect::<Vec<udim>>();
        let pos = Tensor::new(DataType::U32, &[seq_len], reslice::<udim, u8>(&pos));
        // println!("tokens: {tokens:?}");

        let mut x0 = tensor(dt, &[seq_len, d], transfer);
        let e0 = transfer.record();
        let mut x1 = tensor(dt, &[seq_len, d], transfer);
        // `seq_len x hidden_size` -reshape-> `seq_len x (num_kv_head x head_group x head_dim)` -transpose(1,2,0,3)-> `num_kv_head x head_group x seq_len x head_dim` -reshape-> `num_kv_head x (head_group x seq_len) x head_dim`
        let mut x2 = tensor(dt, &[nkvh, head_group * seq_len, dh], transfer);
        let mut qkv = tensor(dt, &[seq_len, d + dkv + dkv], transfer);
        let mut q_att = tensor(dt, &[nh, seq_len, dh], transfer);
        let mut att = tensor(dt, &[nkvh, head_group * seq_len, att_len], transfer);
        let mut gate_up = tensor(dt, &[seq_len, di + di], transfer);
        let e_alloc = transfer.record();

        e0.synchronize();
        // gather(&mut x0.access_mut(), &self.model.embed_tokens(), tokens);
        // println!("gather:\n{}", x0.access());

        e_alloc.synchronize();
        for layer in 0..self.host.num_hidden_layers() {}
    }
}

#[inline]
fn tensor<'a>(dt: DataType, shape: &[udim], stream: &'a Stream) -> Tensor<DevMem<'a>> {
    Tensor::new(
        dt,
        shape,
        DevMem::new(shape.iter().product::<udim>() as usize * dt.size(), stream),
    )
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
        let mut host = PageLockedMemory::new(safetensors.metadata().unwrap().len() as _, ctx);
        safetensors.read_exact(&mut host).unwrap();
        drop(safetensors);
        println!("read to host {:?}", time.elapsed());

        let time = Instant::now();
        let host = Memory::load_safetensors(config, host, false).unwrap();
        println!("load {:?}", time.elapsed());

        let cpy = ctx.stream();

        let time = Instant::now();
        let _transformer = Transformer::new(&host, &cpy);
        println!("build model host: {:?}", time.elapsed());
    });
}
