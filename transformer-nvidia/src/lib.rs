#![cfg(detected_cuda)]

mod cache;
mod kernel;
mod parameters;
mod storage;

use ::half::f16;
use common::{upos, utok};
use cublas::{bindings as cublas_def, cublas};
use cuda::{AsRaw, CudaDataType::half, Stream};
use kernel::{gather, mat_mul, FusedSoftmax, Reform, RmsNormalization, RotaryEmbedding, Swiglu};
use model_parameters::Llama2;
use parameters::{LayersParameters, ModelParameters};
use std::ptr::null_mut;
use storage::DevMem;
use tensor::{slice, udim, DataType, Tensor};

pub use cache::LayerCache;
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
    reform: Reform,
    fused_softmax: FusedSoftmax,
    swiglu: Swiglu,

    logits_dev: Tensor<DevMem<'a>>,
    logits: Vec<f32>,
}

impl Drop for Transformer<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasDestroy_v2(self.cublas));
    }
}

impl<'a> Transformer<'a> {
    pub fn new(host: &'a dyn Llama2, preload_layers: usize, stream: &'a Stream) -> Self {
        let vocab_size = host.vocab_size();
        let load_layers = preload_layers.min(host.num_hidden_layers());

        let mut cublas_handle = null_mut();
        cublas!(cublasCreate_v2(&mut cublas_handle));

        let ctx = stream.ctx();
        let _dev = ctx.dev();
        let block_size = 1024;
        Self {
            model: ModelParameters::new(host, stream),
            layers: LayersParameters::new(load_layers, host, stream),

            cublas: cublas_handle,
            rms_norm: RmsNormalization::new(half, host.hidden_size(), block_size, ctx),
            rotary_embedding: RotaryEmbedding::new(block_size, ctx),
            reform: Reform::new(block_size, 32, ctx),
            fused_softmax: FusedSoftmax::new(half, host.max_position_embeddings(), block_size, ctx),
            swiglu: Swiglu::new(half, block_size, ctx),

            logits_dev: tensor(host.data_type(), &[1, vocab_size as _], stream),
            logits: vec![0.; vocab_size],
            host,
        }
    }

    #[inline]
    pub fn new_cache<'b>(&self, stream: &'b Stream) -> Vec<LayerCache<'b>> {
        LayerCache::new_layers(&*self.host, &stream)
    }

    pub fn update<'b>(
        &mut self,
        tokens: &[utok],
        cache: &[LayerCache],
        pos: upos,
        compute: &Stream,
        transfer: &'b Stream,
    ) -> Tensor<DevMem<'b>> {
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
        for (layer, cache) in cache.iter().enumerate() {
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
            // compute.synchronize();
            // println!("layer {layer} rot q:\n{}", map_tensor(&q));
            // println!("layer {layer} rot k:\n{}", map_tensor(&k));
            let q = q.transpose(&[1, 0, 2]);
            let k = k.transpose(&[1, 0, 2]);
            let v = v.transpose(&[1, 0, 2]);

            let (k_cache, v_cache) = cache.get();
            let k_cat = k_cache.slice(cat_slice);
            let v_cat = v_cache.slice(cat_slice);
            self.reform.launch(&q_att, &q, compute);
            self.reform.launch(&k_cat, &k, compute);
            self.reform.launch(&v_cat, &v, compute);

            let q_att = q_att.clone().reshape(&[nkvh, head_group * seq_len, dh]);
            let k_att = k_cache.slice(att_slice);
            let v_att = v_cache.slice(att_slice);
            // compute.synchronize();
            // println!("layer {layer} q attention:\n{}", map_tensor(&q_att));
            // println!("layer {layer} k attention:\n{}", map_tensor(&k_att));
            // println!("layer {layer} v attention:\n{}", map_tensor(&v_att));

            {
                let k_att = k_att.transpose(&[0, 2, 1]);
                mat_mul(self.cublas, &att, 0., &q_att, &k_att, head_div);
                {
                    let att = att.clone().reshape(&[nh, seq_len, att_len]);
                    // compute.synchronize();
                    // println!("layer {layer} before softmax:\n{}", map_tensor(&att));
                    self.fused_softmax.launch(&att, compute);
                    // compute.synchronize();
                    // println!("layer {layer} after softmax:\n{}", map_tensor(&att));
                }
                mat_mul(self.cublas, &x2, 0., &att, &v_att, 1.);
            }
            self.reform.launch(
                &x1.clone().reshape(&[seq_len, nh, dh]),
                &x2.clone().reshape(&[nh, seq_len, dh]).transpose(&[1, 0, 2]),
                compute,
            );
            // compute.synchronize();
            // println!("layer {layer} after attention:\n{}", map_tensor(&x1));

            let wo = params.self_attn_o_proj.transpose(&[1, 0]);
            mat_mul(self.cublas, &x0, 1., &x1, &wo, 1.);
            // compute.synchronize();
            // println!("layer {layer} o_proj:\n{}", map_tensor(&x0));

            self.rms_norm.launch(
                x1.physical(),
                x0.physical(),
                params.post_attention_layernorm.physical(),
                epsilon,
                d as _,
                compute,
            );
            // compute.synchronize();
            // println!("layer {layer} post norm:\n{}", map_tensor(&x1));

            let w_gate_up = params.mlp_gate_up.transpose(&[1, 0]);
            mat_mul(self.cublas, &gate_up, 0., &x1, &w_gate_up, 1.);
            let mut gate_up = gate_up.split(1, &[di as _, di as _]);
            let up = gate_up.pop().unwrap();
            let gate = gate_up.pop().unwrap();
            // compute.synchronize();
            // println!("layer {layer} gate:\n{}", map_tensor(&gate));
            // println!("layer {layer} up:\n{}", map_tensor(&up));

            self.swiglu.launch(&gate, &up, compute);
            // compute.synchronize();
            // println!("layer {layer} swiglu:\n{}", map_tensor(&gate));

            let mlp_down = params.mlp_down.transpose(&[1, 0]);
            mat_mul(self.cublas, &x0, 1., &gate, &mlp_down, 1.);
            // compute.synchronize();
            // println!("layer {layer} down:\n{}", map_tensor(&x0));
        }

        x0
    }

    pub fn forward(
        &mut self,
        token: utok,
        cache: &[LayerCache],
        pos: upos,
        compute: &Stream,
        transfer: &Stream,
    ) -> &[f32] {
        let x = self.update(&[token], cache, pos, compute, transfer);

        compute.wait_for(&self.model.sync_event);
        self.rms_norm.launch(
            x.physical(),
            x.physical(),
            self.model.model_norm.physical(),
            self.host.rms_norm_eps(),
            self.host.hidden_size(),
            compute,
        );
        // compute.synchronize();
        // println!("pos {pos} model norm:\n{}", map_tensor(&x));

        mat_mul(
            self.cublas,
            &self.logits_dev,
            0.,
            &x,
            &self.model.lm_head.transpose(&[1, 0]),
            1.,
        );
        compute.synchronize();
        cuda::driver!(cuMemcpyDtoH_v2(
            self.logits.as_mut_ptr() as _,
            self.logits_dev.physical().as_raw(),
            self.logits_dev.bytes_size(),
        ));
        // println!("pos {pos} logits:\n{:?}", self.logits);

        match self.host.data_type() {
            DataType::F32 => {}
            DataType::F16 => {
                let ptr = self.logits.as_ptr().cast::<f16>();
                let len = self.host.vocab_size();
                let src = unsafe { std::slice::from_raw_parts(ptr, len) };
                for (dst, src) in self.logits.iter_mut().rev().zip(src.iter().rev()) {
                    *dst = f32::from(*src);
                }
            }
            _ => unreachable!(),
        }
        &self.logits
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
