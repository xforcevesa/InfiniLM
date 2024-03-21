#![cfg(detected_cuda)]

mod kernel;
mod parameters;
mod storage;

#[macro_use]
extern crate log;

use ::half::f16;
use common::utok;
use cublas::Cublas;
use cuda::{AsRaw, CudaDataType::half, Stream};
use kernel::{gather, mat_mul, FusedSoftmax, Reform, RmsNormalization, RotaryEmbedding, Swiglu};
use parameters::{LayersParameters, ModelParameters};
use storage::Storage;
use tensor::{slice, udim, DataType, Tensor};
use transformer::SampleArgs;

pub type Request<'a, 'b, Id> = transformer::Request<'a, Id, Storage<'b>>;
pub type LayerCache<'a> = transformer::LayerCache<Storage<'a>>;
pub use transformer::{Llama2, Memory};
pub extern crate cuda;

pub struct Transformer<'ctx> {
    host: &'ctx dyn Llama2,
    model: ModelParameters<'ctx>,
    layers: LayersParameters<'ctx>,
    cublas: cublas::Cublas<'ctx>,
    rms_norm: RmsNormalization<'ctx>,
    rotary_embedding: RotaryEmbedding<'ctx>,
    reform: Reform<'ctx>,
    fused_softmax: FusedSoftmax<'ctx>,
    swiglu: Swiglu<'ctx>,
}

impl<'ctx> Transformer<'ctx> {
    pub fn new(host: &'ctx dyn Llama2, preload_layers: usize, stream: &'ctx Stream) -> Self {
        let load_layers = preload_layers.min(host.num_hidden_layers());
        let ctx = stream.ctx();
        let dev = ctx.dev();
        let (block_size, _) = dev.max_block_dims();
        Self {
            model: ModelParameters::new(host, stream),
            layers: LayersParameters::new(load_layers, host, stream),
            cublas: Cublas::new(ctx),
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
        LayerCache::new_layers(self.host, |dt, shape| tensor(dt, shape, stream))
    }

    pub fn decode<Id>(
        &mut self,
        mut requests: Vec<Request<Id>>,
        sample: &SampleArgs,
        compute: &Stream<'ctx>,
        transfer: &Stream<'ctx>,
    ) -> Vec<(Id, utok)> {
        // 归拢所有纯解码的请求到前面，减少解码 batching 的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);

        // println!("tokens:");
        // for request in requests.iter() {
        //     println!(
        //         "{:?}: {:?}",
        //         request.tokens,
        //         request.pos..request.pos + request.tokens.len() as upos
        //     );
        // }

        // `nt` for number of tokens
        let (nt, max_seq_len, max_att_len) =
            requests
                .iter()
                .fold((0, 0, 0), |(nt, max_seq, max_att), r| {
                    let seq_len = r.seq_len();
                    let att_len = r.att_len();
                    (nt + seq_len, max_seq.max(seq_len), max_att.max(att_len))
                });

        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let di = self.host.intermediate_size() as udim;
        let voc = self.host.vocab_size() as udim;
        let dt = self.host.data_type();
        let epsilon = self.host.rms_norm_eps();
        let theta = self.host.rope_theta();
        let mut pos = tensor(DataType::U32, &[nt], transfer);
        let mut pos_ = Vec::<u32>::with_capacity(nt as usize);
        for request in requests.iter() {
            pos_.extend(request.pos()..request.att_len());
        }
        pos.physical_mut().copy_in_async(&pos_, transfer);

        let mut x0 = tensor(dt, &[nt, d], compute);
        let mut x1 = tensor(dt, &[nt, d], transfer);
        let mut qkv = tensor(dt, &[nt, d + dkv + dkv], transfer);
        let mut q_buf = Storage::new((nh * max_seq_len * dh) as usize * dt.size(), transfer);
        let mut att_buf = Storage::new(
            (nh * max_seq_len * max_att_len) as usize * dt.size(),
            transfer,
        );
        let mut gate_up = tensor(dt, &[nt, di + di], transfer);
        let e_alloc = transfer.record();

        let tokens = requests.iter().flat_map(Request::tokens).copied();
        gather(&mut x0, &self.host.embed_tokens(), tokens, compute);
        // compute.synchronize();
        // println!("gather:\n{}", map_tensor(&x0));

        self.cublas.set_stream(compute);
        compute.wait_for(&e_alloc);
        for layer in 0..self.host.num_hidden_layers() {
            self.layers.load(layer, self.host, transfer);
            let params = self.layers.sync(layer, compute);

            self.rms_norm
                .launch(&mut x1, &x0, &params.input_layernorm, epsilon, compute);
            // compute.synchronize();
            // println!("layer {layer} input norm:\n{}", map_tensor(&x1));

            mat_mul(&self.cublas, &mut qkv, 0., &x1, &params.w_qkv, 1.);
            let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
            let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
            let mut k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
            let mut q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
            // compute.synchronize();
            // println!("layer {layer} q:\n{}", map_tensor(&q));
            // println!("layer {layer} k:\n{}", map_tensor(&k));
            // println!("layer {layer} v:\n{}", map_tensor(&v));

            self.rotary_embedding.launch(&mut q, &pos, theta, compute);
            self.rotary_embedding.launch(&mut k, &pos, theta, compute);
            // compute.synchronize();
            // println!("layer {layer} rot q:\n{}", map_tensor(&q));
            // println!("layer {layer} rot k:\n{}", map_tensor(&k));

            let q = q.as_ref().transpose(&[1, 0, 2]);
            let k = k.as_ref().transpose(&[1, 0, 2]);
            let v = v.as_ref().transpose(&[1, 0, 2]);
            let mut o = x1.as_mut().reshape(&[nt, nh, dh]).transpose(&[1, 0, 2]);

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

                let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut *q_buf);
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

                let mut att = Tensor::new(dt, shape_att0, &mut *att_buf);
                mat_mul(&self.cublas, &mut att, 0., &q_att, &k_att, head_div);
                let mut att = att.reshape(shape_att1);
                self.fused_softmax.launch(&mut att, compute);
                let mut x2 = q_att;
                let att = att.reshape(shape_att0);
                mat_mul(&self.cublas, &mut x2, 0., &att, &v_att, 1.);

                self.reform
                    .launch(&mut o, &x2.reshape(&[nh, seq_len, dh]), compute);
                // println!("layer {layer} after attention:\n{}", o);
            }

            mat_mul(&self.cublas, &mut x0, 1., &x1, &params.self_attn_o_proj, 1.);
            // compute.synchronize();
            // println!("layer {layer} o_proj:\n{}", map_tensor(&x0));

            self.rms_norm.launch(
                &mut x1,
                &x0,
                &params.post_attention_layernorm,
                epsilon,
                compute,
            );
            // compute.synchronize();
            // println!("layer {layer} post norm:\n{}", map_tensor(&x1));

            mat_mul(&self.cublas, &mut gate_up, 0., &x1, &params.mlp_gate_up, 1.);
            let mut gate_up = gate_up.split(1, &[di as _, di as _]);
            let up = gate_up.pop().unwrap();
            let mut gate = gate_up.pop().unwrap();
            // compute.synchronize();
            // println!("layer {layer} gate:\n{}", map_tensor(&gate));
            // println!("layer {layer} up:\n{}", map_tensor(&up));

            self.swiglu.launch(&mut gate, &up, compute);
            // compute.synchronize();
            // println!("layer {layer} swiglu:\n{}", map_tensor(&gate));

            mat_mul(&self.cublas, &mut x0, 1., &gate, &params.mlp_down, 1.);
            // compute.synchronize();
            // println!("layer {layer} down:\n{}", map_tensor(&x0));
        }

        let (head, others) = requests.split_first().unwrap();
        if !head.decode() {
            return vec![];
        }

        let tokens = {
            let begin = head.seq_len() as usize;
            let mut i = begin;
            let mut j = begin;
            let buf = unsafe { x0.physical().as_raw() };
            let len = d as usize * dt.size();
            for r in others {
                i += r.seq_len() as usize;
                j += 1;
                if r.decode() && i > j {
                    cuda::driver!(cuMemcpyDtoDAsync_v2(
                        buf + ((j - 1) * len) as CUdeviceptr,
                        buf + ((i - 1) * len) as CUdeviceptr,
                        len,
                        compute.as_raw()
                    ));
                }
            }
            let begin = begin as udim - 1;
            let len = j as udim - begin;
            slice![from begin, take len]
        };

        let mut logits_dev = tensor(dt, &[tokens.len, voc], compute);
        let mut x = x0.slice(&[tokens, slice![all]]);
        // compute.synchronize();
        // println!("decode slice:\n{}", map_tensor(&x));

        compute.wait_for(&self.model.sync_event);
        // 复制一个 x 以实现原地归一化
        let x_ = unsafe { x.as_ref().map_physical(|u| u.borrow()) };
        self.rms_norm
            .launch(&mut x, &x_, &self.model.model_norm, epsilon, compute);
        // compute.synchronize();
        // println!("model norm:\n{}", map_tensor(&x));

        mat_mul(
            &self.cublas,
            &mut logits_dev,
            0.,
            &x,
            &self.model.lm_head,
            1.,
        );

        let mut logits = vec![f16::ZERO; logits_dev.size()];
        compute.synchronize();
        logits_dev.physical().copy_out(&mut logits);
        requests
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                (
                    r.id(),
                    sample.random(&mut logits[i * voc as usize..][..voc as usize]),
                )
            })
            .collect()
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
