#![cfg(detected_cuda)]

mod kernel;
mod parameters;
mod sample;
mod storage;

#[macro_use]
extern crate log;

use common::utok;
use cublas::{Cublas, CublasSpore};
use cuda::{AsRaw, ContextResource, ContextSpore, CudaDataType::half, Stream};
use kernel::{gather, mat_mul, FusedSoftmax, Reform, RmsNormalization, RotaryEmbedding, Swiglu};
use parameters::{LayerParameter, LayersParameters, ModelParameters};
use std::sync::Mutex;
use storage::Storage;
use tensor::{slice, udim, DataType, Tensor};
use transformer::{pos, LayerBuffer, Sample as _, SampleArgs};

pub type Request<'a, 'b, Id> = transformer::Request<'a, Id, Storage<'b>>;
pub type LayerCache<'a> = transformer::LayerCache<Storage<'a>>;
pub use sample::Sample;
pub use transformer::{Llama2, Memory};
pub extern crate cuda;

pub struct Transformer<'ctx> {
    host: Box<dyn Llama2>,
    model: ModelParameters,
    layers: Mutex<LayersParameters<'ctx>>,
    cublas: CublasSpore,
    rms_norm: RmsNormalization<'ctx>,
    rotary_embedding: RotaryEmbedding<'ctx>,
    reform: Reform<'ctx>,
    fused_softmax: FusedSoftmax<'ctx>,
    swiglu: Swiglu<'ctx>,
}

impl<'ctx> Transformer<'ctx> {
    pub fn new(host: Box<dyn Llama2>, preload_layers: usize, stream: &'ctx Stream) -> Self {
        let load_layers = preload_layers.min(host.num_hidden_layers());
        let ctx = stream.ctx();
        let dev = ctx.dev();
        let (block_size, _) = dev.max_block_dims();
        Self {
            model: ModelParameters::new(&*host, stream),
            layers: Mutex::new(LayersParameters::new(load_layers, &*host, stream)),
            cublas: Cublas::new(ctx).sporulate(),
            rms_norm: RmsNormalization::new(half, host.hidden_size(), block_size, ctx),
            rotary_embedding: RotaryEmbedding::new(block_size, ctx),
            reform: Reform::new(block_size, 32, ctx),
            fused_softmax: FusedSoftmax::new(half, host.max_position_embeddings(), block_size, ctx),
            swiglu: Swiglu::new(half, block_size, ctx),
            host,
        }
    }

    #[inline]
    pub fn new_cache<'b>(&self, stream: &'b Stream) -> Vec<LayerCache<'b>> {
        LayerCache::new_layers(&*self.host, |dt, shape| tensor(dt, shape, stream))
    }

    pub fn decode<Id>(
        &self,
        mut requests: Vec<Request<Id>>,
        sample: &SampleArgs,
        compute: &Stream<'ctx>,
        transfer: &Stream<'ctx>,
    ) -> Vec<(Id, utok)> {
        let ctx = compute.ctx();
        unsafe { self.cublas.sprout(ctx) }.set_stream(compute);
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);
        // 生成词嵌入并预分配空间
        let mut x0 = self.token_embed(&requests, compute);
        let mut x1 = tensor(x0.data_type(), x0.shape(), transfer);
        let mut buf =
            LayerBuffer::alloc(&*self.host, &requests, |size| Storage::new(size, transfer));
        // 生成位置张量
        let nt = x0.shape()[0]; // `nt` for number of tokens
        let pos_ = pos(&requests, nt);
        let mut pos = tensor(DataType::U32, &[nt], transfer);
        pos.physical_mut().copy_in_async(&pos_, transfer);
        // 推理
        compute.wait_for(&transfer.record());
        {
            // 层参数滚动加载是有状态的，必须由一个控制流独占。其他逻辑无状态，可以多流并发
            let mut layers = self.layers.lock().unwrap();
            for layer in 0..self.host.num_hidden_layers() {
                let params = {
                    layers.load(layer, &*self.host, transfer);
                    layers.sync(layer, compute)
                };

                let (q, k, v) = self.before_att(params, &x0, &mut x1, &mut buf.qkv, &pos, compute);
                let o = &mut x1;
                #[rustfmt::skip]
                self.attention(layer, &mut requests, q, k, v, o, &mut buf.q_buf, &mut buf. att_buf, compute);
                self.after_att(params, &mut x0, &mut x1, &mut buf.gate_up, compute);
            }
        }
        // 解码
        if requests[0].decode() {
            let x = self.move_decode(&requests, x0, compute);
            let requests = requests.into_iter().map(Request::id).collect::<Vec<_>>();
            Sample.sample(sample, requests, self.logits(x, compute))
        } else {
            vec![]
        }
    }

    fn token_embed<Id>(
        &self,
        requests: &[Request<Id>],
        compute: &Stream<'ctx>,
    ) -> Tensor<Storage<'ctx>> {
        let dt = self.host.data_type();
        let nt = requests.iter().map(Request::seq_len).sum::<udim>();
        let d = self.host.hidden_size() as udim;

        let mut x0 = tensor(dt, &[nt, d], compute);

        let tokens = requests.iter().flat_map(Request::tokens).copied();
        gather(&mut x0, &self.host.embed_tokens(), tokens, compute);
        // compute.synchronize();
        // println!("gather:\n{}", map_tensor(&x0));

        x0
    }

    fn before_att(
        &self,
        params: &LayerParameter,
        x0: &Tensor<Storage>,
        x1: &mut Tensor<Storage>,
        qkv: &mut Tensor<Storage<'ctx>>,
        pos: &Tensor<Storage>,
        compute: &Stream,
    ) -> (
        Tensor<Storage<'ctx>>,
        Tensor<Storage<'ctx>>,
        Tensor<Storage<'ctx>>,
    ) {
        let nt = x0.shape()[0];
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let epsilon = self.host.rms_norm_eps();
        let theta = self.host.rope_theta();
        let cublas = unsafe { self.cublas.sprout(compute.ctx()) };

        self.rms_norm
            .launch(x1, x0, &params.input_layernorm, epsilon, compute);
        // compute.synchronize();
        // println!("layer {layer} input norm:\n{}", map_tensor(&x1));

        mat_mul(&cublas, qkv, 0., x1, &params.w_qkv, 1.);
        let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
        let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
        // compute.synchronize();
        // println!("layer {layer} q:\n{}", map_tensor(&q));
        // println!("layer {layer} k:\n{}", map_tensor(&k));
        // println!("layer {layer} v:\n{}", map_tensor(&v));

        self.rotary_embedding.launch(&mut q, pos, theta, compute);
        self.rotary_embedding.launch(&mut k, pos, theta, compute);
        // compute.synchronize();
        // println!("layer {layer} rot q:\n{}", map_tensor(&q));
        // println!("layer {layer} rot k:\n{}", map_tensor(&k));

        (q, k, v)
    }

    fn attention<Id>(
        &self,
        layer: usize,
        requests: &mut [Request<Id>],
        q: Tensor<Storage>,
        k: Tensor<Storage>,
        v: Tensor<Storage>,
        o: &mut Tensor<Storage>,
        q_buf: &mut Storage,
        att_buf: &mut Storage,
        compute: &Stream,
    ) {
        let dt = self.host.data_type();
        let nt = o.shape()[0];
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let cublas = unsafe { self.cublas.sprout(compute.ctx()) };

        let q = q.as_ref().transpose(&[1, 0, 2]);
        let k = k.as_ref().transpose(&[1, 0, 2]);
        let v = v.as_ref().transpose(&[1, 0, 2]);
        let mut o = o.as_mut().reshape(&[nt, nh, dh]).transpose(&[1, 0, 2]);

        let q = unsafe { q.map_physical(|u| &**u) };
        let k = unsafe { k.map_physical(|u| &**u) };
        let v = unsafe { v.map_physical(|u| &**u) };

        let mut req = 0;
        for r in requests.iter_mut() {
            let pos = r.pos();
            let seq_len = r.seq_len();
            let att_len = r.att_len();

            let req_slice = &[slice![all], slice![from req, take seq_len], slice![all]];
            let cat_slice = &[slice![all], slice![from pos, take seq_len], slice![all]];
            let att_slice = &[slice![all], slice![from   0, take att_len], slice![all]];
            req += seq_len;

            let q = q.clone().slice(req_slice);
            let k = k.clone().slice(req_slice);
            let v = v.clone().slice(req_slice);
            let o = o.as_mut().slice(req_slice);
            let mut o = unsafe { o.map_physical(|u| &mut ***u) };

            let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut **q_buf);
            let (k_cache, v_cache) = r.cache(layer);
            let k_cat = k_cache.as_mut().slice(cat_slice);
            let v_cat = v_cache.as_mut().slice(cat_slice);
            let mut k_cat = unsafe { k_cat.map_physical(|u| &mut **u) };
            let mut v_cat = unsafe { v_cat.map_physical(|u| &mut **u) };
            self.reform.launch(&mut q_att, &q, compute);
            self.reform.launch(&mut k_cat, &k, compute);
            self.reform.launch(&mut v_cat, &v, compute);

            let q_att = q_att.reshape(&[nkvh, head_group * seq_len, dh]);
            let k_att = k_cache.as_ref().slice(att_slice).transpose(&[0, 2, 1]);
            let v_att = v_cache.as_ref().slice(att_slice);
            let k_att = unsafe { k_att.map_physical(|u| &**u) };
            let v_att = unsafe { v_att.map_physical(|u| &**u) };
            // println!("layer {layer} q attention:\n{}", q_att);
            // println!("layer {layer} k attention:\n{}", k_att.access());
            // println!("layer {layer} v attention:\n{}", v_att.access());

            let shape_att0 = &[nkvh, head_group * seq_len, att_len];
            let shape_att1 = &[nkvh * head_group, seq_len, att_len];

            let mut att = Tensor::new(dt, shape_att0, &mut **att_buf);
            mat_mul(&cublas, &mut att, 0., &q_att, &k_att, head_div);
            let mut att = att.reshape(shape_att1);
            self.fused_softmax.launch(&mut att, compute);
            let mut x2 = q_att;
            let att = att.reshape(shape_att0);
            mat_mul(&cublas, &mut x2, 0., &att, &v_att, 1.);

            self.reform
                .launch(&mut o, &x2.reshape(&[nh, seq_len, dh]), compute);
            // println!("layer {layer} after attention:\n{}", o);
        }
    }

    fn after_att(
        &self,
        params: &LayerParameter,
        x0: &mut Tensor<Storage>,
        x1: &mut Tensor<Storage>,
        gate_up: &mut Tensor<Storage>,
        compute: &Stream,
    ) {
        let di = self.host.intermediate_size() as udim;
        let epsilon = self.host.rms_norm_eps();
        let cublas = unsafe { self.cublas.sprout(compute.ctx()) };

        mat_mul(&cublas, x0, 1., x1, &params.self_attn_o_proj, 1.);
        // compute.synchronize();
        // println!("layer {layer} o_proj:\n{}", map_tensor(&x0));

        self.rms_norm
            .launch(x1, x0, &params.post_attention_layernorm, epsilon, compute);
        // compute.synchronize();
        // println!("layer {layer} post norm:\n{}", map_tensor(&x1));

        mat_mul(&cublas, gate_up, 0., x1, &params.mlp_gate_up, 1.);
        let mut gate_up = gate_up.split(1, &[di as _, di as _]);
        let up = gate_up.pop().unwrap();
        let mut gate = gate_up.pop().unwrap();
        // compute.synchronize();
        // println!("layer {layer} gate:\n{}", map_tensor(&gate));
        // println!("layer {layer} up:\n{}", map_tensor(&up));

        self.swiglu.launch(&mut gate, &up, compute);
        // compute.synchronize();
        // println!("layer {layer} swiglu:\n{}", map_tensor(&gate));

        mat_mul(&cublas, x0, 1., &gate, &params.mlp_down, 1.);
        // compute.synchronize();
        // println!("layer {layer} down:\n{}", map_tensor(&x0));
    }

    fn move_decode<Id>(
        &self,
        requests: &[Request<Id>],
        x0: Tensor<Storage<'ctx>>,
        compute: &Stream,
    ) -> Tensor<Storage<'ctx>> {
        let buf = unsafe { x0.physical().as_raw() };
        let len = self.host.hidden_size() * self.host.data_type().size();

        let (head, others) = requests.split_first().unwrap();
        let begin = head.seq_len() as usize - 1;

        let mut src = begin;
        let mut dst = begin;
        for r in others {
            src += r.seq_len() as usize;
            if r.decode() {
                dst += 1;
                if dst < src {
                    cuda::driver!(cuMemcpyDtoDAsync_v2(
                        buf + (dst * len) as CUdeviceptr,
                        buf + (src * len) as CUdeviceptr,
                        len,
                        compute.as_raw()
                    ));
                }
            }
        }

        x0.slice(&[slice![from begin, until dst + 1], slice![all]])
    }

    fn logits(&self, mut x: Tensor<Storage>, compute: &Stream<'ctx>) -> Tensor<Storage<'ctx>> {
        let dt = self.host.data_type();
        let voc = self.host.vocab_size() as udim;
        let epsilon = self.host.rms_norm_eps();
        let cublas = unsafe { self.cublas.sprout(compute.ctx()) };

        let mut logits = tensor(dt, &[x.shape()[0], voc], compute);

        let (model_norm, lm_head) = unsafe { self.model.release(compute) };
        // 复制一个 x 以实现原地归一化
        let x_ = unsafe { x.as_ref().map_physical(|u| u.borrow()) };
        self.rms_norm
            .launch(&mut x, &x_, &model_norm, epsilon, compute);
        // compute.synchronize();
        // println!("model norm:\n{}", map_tensor(&x));

        mat_mul(&cublas, &mut logits, 0., &x, &lm_head, 1.);
        // compute.synchronize();
        // println!("model norm:\n{}", map_tensor(&logits));

        logits
    }
}

#[inline]
fn tensor<'ctx>(dt: DataType, shape: &[udim], stream: &Stream<'ctx>) -> Tensor<Storage<'ctx>> {
    Tensor::new(
        dt,
        shape,
        Storage::new(shape.iter().product::<udim>() as usize * dt.size(), stream),
    )
}

#[allow(unused)]
fn map_tensor(tensor: &Tensor<Storage>) -> Tensor<Vec<u8>> {
    unsafe {
        tensor.as_ref().map_physical(|dev| {
            let mut buf = vec![0; dev.len()];
            dev.copy_out(&mut buf);
            buf
        })
    }
}
