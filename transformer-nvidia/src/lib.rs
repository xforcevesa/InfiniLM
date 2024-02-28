#![cfg(detected_cuda)]

mod kernel;
mod parameters;
mod storage;

use common::{upos, utok};
use cublas::{bindings as cublas_def, cublas};
use cuda::{AsRaw, CudaDataType::half, Stream};
use kernel::{gather, mat_mul, RmsNormalization, RotaryEmbedding};
use model_parameters::Llama2;
use parameters::{LayersParameters, ModelParameters};
use std::ptr::null_mut;
use storage::DevMem;
use tensor::{slice, udim, DataType, Tensor};

pub use storage::PageLockedMemory;

pub extern crate cuda;
pub extern crate model_parameters;

pub struct Transformer<'a> {
    host: &'a dyn Llama2,
    model: ModelParameters<'a>,
    layers: LayersParameters<'a>,

    cublas: cublas_def::cublasHandle_t,
    rms_norm: RmsNormalization,
    rotary_embedding: RotaryEmbedding,
}

impl Drop for Transformer<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasDestroy_v2(self.cublas));
    }
}

impl<'a> Transformer<'a> {
    pub fn new(host: &'a dyn Llama2, stream: &'a Stream) -> Self {
        let d = host.hidden_size();
        let mut cublas_handle = null_mut();
        cublas!(cublasCreate_v2(&mut cublas_handle));
        Self {
            host,
            model: ModelParameters::new(host, stream),
            layers: LayersParameters::new(3, host, stream),

            cublas: cublas_handle,
            rms_norm: RmsNormalization::new(half, d, 1024, stream.ctx()),
            rotary_embedding: RotaryEmbedding::new(1024, stream.ctx()),
        }
    }

    pub fn update(
        &mut self,
        tokens: &[utok],
        // cache: &mut [LayerCache],
        pos: upos,
        compute: &Stream,
        transfer: &Stream,
    ) {
        let seq_len = tokens.len() as udim;
        let d = self.host.hidden_size() as udim;
        let nlayer = self.host.num_hidden_layers();
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
        let pos = DevMem::from_slice(&(pos..pos + seq_len).collect::<Vec<udim>>(), transfer);
        let pos = Tensor::new(DataType::U32, &[seq_len], pos);
        // println!("tokens: {tokens:?}");

        let x0 = tensor(dt, &[seq_len, d], transfer);
        let e_alloc_x0 = transfer.record();
        let x1 = tensor(dt, &[seq_len, d], transfer);
        // `seq_len x hidden_size` -reshape-> `seq_len x (num_kv_head x head_group x head_dim)` -transpose(1,2,0,3)-> `num_kv_head x head_group x seq_len x head_dim` -reshape-> `num_kv_head x (head_group x seq_len) x head_dim`
        let x2 = tensor(dt, &[nkvh, head_group * seq_len, dh], transfer);
        let qkv = tensor(dt, &[seq_len, d + dkv + dkv], transfer);
        let q_att = tensor(dt, &[nh, seq_len, dh], transfer);
        let att = tensor(dt, &[nkvh, head_group * seq_len, att_len], transfer);
        let gate_up = tensor(dt, &[seq_len, di + di], transfer);
        let e_alloc = transfer.record();

        compute.wait_for(&e_alloc_x0);
        gather(&x0, &self.host.embed_tokens(), tokens, compute);
        // compute.synchronize();
        // println!("gather:\n{}", map_tensor(&x0));

        cublas!(cublasSetStream_v2(self.cublas, compute.as_raw() as _));
        compute.wait_for(&e_alloc);
        for layer in 0..nlayer {
            self.layers.load(layer, self.host, transfer);
            let params = self.layers.sync(layer, compute);

            self.rms_norm.launch(
                x1.physical(),
                x0.physical(),
                params.input_layernorm.physical(),
                epsilon,
                d as usize,
                compute,
            );
            // compute.synchronize();
            // println!("layer {layer} input norm:\n{}", map_tensor(&x1));
            let w_qkv = params.w_qkv.transpose(&[1, 0]);
            mat_mul(self.cublas, &qkv, 0., &x1, &w_qkv, 1.);
            let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
            let v = qkv.pop().unwrap().reshape(&[seq_len, nkvh, dh]);
            let k = qkv.pop().unwrap().reshape(&[seq_len, nkvh, dh]);
            let q = qkv.pop().unwrap().reshape(&[seq_len, nh, dh]);
            // compute.synchronize();
            // println!("layer {layer} q:\n{}", map_tensor(&q));
            // println!("layer {layer} k:\n{}", map_tensor(&k));
            // println!("layer {layer} v:\n{}", map_tensor(&v));
            self.rotary_embedding.launch(&q, &pos, theta, compute);
            self.rotary_embedding.launch(&k, &pos, theta, compute);
            println!("layer {layer} rot q:\n{}", map_tensor(&q));
            println!("layer {layer} rot k:\n{}", map_tensor(&k));
            let q = q.transpose(&[1, 0, 2]);
            let k = k.transpose(&[1, 0, 2]);
            let v = v.transpose(&[1, 0, 2]);

            // let (k_cache, v_cache) = cache.get();
            // let mut k_cat = k_cache.slice(cat_slice);
            // let mut v_cat = v_cache.slice(cat_slice);
        }
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

#[allow(unused)]
fn map_tensor(tensor: &Tensor<DevMem>) -> Tensor<Vec<u8>> {
    unsafe {
        tensor.map_physical(|dev| {
            let len = dev.len();
            let mut buf = vec![0; len];
            cuda::driver!(cuMemcpyDtoH_v2(buf.as_mut_ptr() as _, dev.as_raw(), len));
            buf
        })
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
