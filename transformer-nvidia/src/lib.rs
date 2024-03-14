#![cfg(detected_cuda)]

mod kernel;
mod parameters;
mod storage;

#[macro_use]
extern crate log;

use cublas::Cublas;
use cuda::{AsRaw, CudaDataType::half, Stream};
use kernel::{gather, mat_mul, FusedSoftmax, Reform, RmsNormalization, RotaryEmbedding, Swiglu};
use parameters::{LayersParameters, ModelParameters};
use storage::Storage;
use tensor::{slice, udim, DataType, PhysicalCell, Tensor};

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
        mut requests: Vec<Request<'_, 'ctx, Id>>,
        compute: &Stream<'ctx>,
        transfer: &Stream<'ctx>,
    ) -> (Vec<Id>, Tensor<Vec<u8>>) {
        requests.sort_unstable_by_key(|t| t.tokens.len());

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
            pos_.extend(request.pos..request.att_len());
        }
        pos.access_mut()
            .physical_mut()
            .copy_in_async(&pos_, transfer);

        let mut x0 = tensor(dt, &[nt, d], transfer);
        let e_alloc_x0 = transfer.record();
        let mut x1 = tensor(dt, &[nt, d], transfer);
        let qkv = tensor(dt, &[nt, d + dkv + dkv], transfer);
        let q_buf = Storage::new((nh * max_seq_len * dh) as usize * dt.size(), transfer);
        let att_buf = Storage::new(
            (nkvh * head_group * max_seq_len * max_att_len) as usize * dt.size(),
            transfer,
        );
        //                         `num_token x hidden_size`
        // -|reshape|------------> `num_token x (num_kv_head x head_group x head_dim)`
        // -|transpose(1,2,0,3)|-> `num_kv_head x head_group x num_token x head_dim`
        // -|reshape|------------> `num_kv_head x (head_group x num_token) x head_dim`
        let x2 = tensor(dt, &[nkvh, head_group * nt, dh], transfer);
        let gate_up = tensor(dt, &[nt, di + di], transfer);
        let e_alloc = transfer.record();

        compute.wait_for(&e_alloc_x0);
        gather(
            x0.access_mut(),
            &self.host.embed_tokens(),
            requests.iter().map(|r| r.tokens),
            compute,
        );
        // compute.synchronize();
        // println!("gather:\n{}", map_tensor(&x0));

        self.cublas.set_stream(compute);
        compute.wait_for(&e_alloc);
        for layer in 0..self.host.num_hidden_layers() {
            self.layers.load(layer, self.host, transfer);
            let params = self.layers.sync(layer, compute);

            self.rms_norm.launch(
                x1.access_mut(),
                &x0.access(),
                &params.input_layernorm.access(),
                epsilon,
                compute,
            );
            // compute.synchronize();
            // println!("layer {layer} input norm:\n{}", map_tensor(&x1));
            let w_qkv = params.w_qkv.clone().transpose(&[1, 0]);
            mat_mul(
                &self.cublas,
                &qkv.access(),
                0.,
                &x1.access(),
                &w_qkv.access(),
                1.,
            );
            let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
            let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
            let k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
            let q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
            // compute.synchronize();
            // println!("layer {layer} q:\n{}", map_tensor(&q));
            // println!("layer {layer} k:\n{}", map_tensor(&k));
            // println!("layer {layer} v:\n{}", map_tensor(&v));
            self.rotary_embedding
                .launch(&q.access(), &pos.access(), theta, compute);
            self.rotary_embedding
                .launch(&k.access(), &pos.access(), theta, compute);
            // compute.synchronize();
            // println!("layer {layer} rot q:\n{}", map_tensor(&q));
            // println!("layer {layer} rot k:\n{}", map_tensor(&k));
            let q = q.transpose(&[1, 0, 2]);
            let k = k.transpose(&[1, 0, 2]);
            let v = v.transpose(&[1, 0, 2]);

            let mut req = 0;
            for r in requests.iter_mut() {
                let pos = r.pos;
                let seq_len = r.seq_len();
                let att_len = r.att_len();

                let req_slice = &[slice![all], slice![from req, take seq_len], slice![all]];
                let cat_slice = &[slice![all], slice![from pos, take seq_len], slice![all]];
                let att_slice = &[slice![all], slice![from   0, take att_len], slice![all]];
                req += seq_len;

                let q = q.clone().slice(req_slice);
                let k = k.clone().slice(req_slice);
                let v = v.clone().slice(req_slice);

                let (k_cache, v_cache) = r.cache[layer].get();
                let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], q_buf.clone());
                let mut k_cat = k_cache.clone().slice(cat_slice);
                let mut v_cat = v_cache.clone().slice(cat_slice);
                self.reform.launch(q_att.access_mut(), &q.access(), compute);
                self.reform.launch(k_cat.access_mut(), &k.access(), compute);
                self.reform.launch(v_cat.access_mut(), &v.access(), compute);

                let q_att = q_att.reshape(&[nkvh, head_group * seq_len, dh]);
                let k_att = k_cache.clone().slice(att_slice);
                let v_att = v_cache.clone().slice(att_slice);
                // println!("layer {layer} q attention:\n{}", map_tensor(&q_att));
                // println!("layer {layer} k attention:\n{}", map_tensor(&k_att));
                // println!("layer {layer} v attention:\n{}", map_tensor(&v_att));

                let mut att =
                    Tensor::new(dt, &[nkvh, head_group * seq_len, att_len], att_buf.clone());
                {
                    let k_att = k_att.transpose(&[0, 2, 1]);
                    mat_mul(
                        &self.cublas,
                        &att.access(),
                        0.,
                        &q_att.access(),
                        &k_att.access(),
                        head_div,
                    );
                    att = att.reshape(&[nh, seq_len, att_len]);
                    // compute.synchronize();
                    // println!("layer {layer} before softmax:\n{}", map_tensor(&att));
                    self.fused_softmax.launch(&att.access(), compute);
                    // compute.synchronize();
                    // println!("layer {layer} after softmax:\n{}", map_tensor(&att));
                    att = att.reshape(&[nkvh, head_group * seq_len, att_len]);
                    {
                        mat_mul(
                            &self.cublas,
                            &x2.access(),
                            0.,
                            &att.access(),
                            &v_att.access(),
                            1.,
                        );
                        self.reform.launch(
                            x1.clone().reshape(&[seq_len, nh, dh]).access_mut(),
                            &x2.clone()
                                .reshape(&[nh, seq_len, dh])
                                .transpose(&[1, 0, 2])
                                .access(),
                            compute,
                        );
                    }
                    // compute.synchronize();
                    // println!("layer {layer} after attention:\n{}", map_tensor(&x1));
                }
            }

            let wo = params.self_attn_o_proj.clone().transpose(&[1, 0]);
            mat_mul(
                &self.cublas,
                &x0.access(),
                1.,
                &x1.access(),
                &wo.access(),
                1.,
            );
            // compute.synchronize();
            // println!("layer {layer} o_proj:\n{}", map_tensor(&x0));

            self.rms_norm.launch(
                x1.access_mut(),
                &x0.access(),
                &params.post_attention_layernorm.access(),
                epsilon,
                compute,
            );
            // compute.synchronize();
            // println!("layer {layer} post norm:\n{}", map_tensor(&x1));

            let w_gate_up = params.mlp_gate_up.clone().transpose(&[1, 0]);
            mat_mul(
                &self.cublas,
                &gate_up.access(),
                0.,
                &x1.access(),
                &w_gate_up.access(),
                1.,
            );
            let mut gate_up = gate_up.split(1, &[di as _, di as _]);
            let up = gate_up.pop().unwrap();
            let gate = gate_up.pop().unwrap();
            // compute.synchronize();
            // println!("layer {layer} gate:\n{}", map_tensor(&gate));
            // println!("layer {layer} up:\n{}", map_tensor(&up));

            self.swiglu.launch(&gate.access(), &up.access(), compute);
            // compute.synchronize();
            // println!("layer {layer} swiglu:\n{}", map_tensor(&gate));

            let mlp_down = params.mlp_down.clone().transpose(&[1, 0]);
            mat_mul(
                &self.cublas,
                &x0.access(),
                1.,
                &gate.access(),
                &mlp_down.access(),
                1.,
            );
            compute.synchronize();
            // println!("layer {layer} down:\n{}", map_tensor(&x0));
        }

        let tokens = {
            let (head, others) = requests.split_first().unwrap();
            let begin = head.tokens.len();
            let mut i = begin;
            let mut j = begin;
            let buf = unsafe { x0.access().physical().as_raw() };
            let len = d as usize * dt.size();
            for r in others {
                i += r.tokens.len();
                j += 1;
                if i > j {
                    cuda::driver!(cuMemcpyDtoDAsync_v2(
                        buf + ((j - 1) * len) as cuda::bindings::CUdeviceptr,
                        buf + ((i - 1) * len) as cuda::bindings::CUdeviceptr,
                        len,
                        compute.as_raw()
                    ));
                }
            }
            let begin = begin as udim - 1;
            let len = j as udim - begin;
            slice![from begin, take len]
        };

        let logits_dev = tensor(dt, &[tokens.len, voc], compute);
        let mut x = x0.slice(&[tokens, slice![all]]);
        // compute.synchronize();
        // println!("decode slice:\n{}", map_tensor(&x));

        compute.wait_for(&self.model.sync_event);
        let x_ = x.clone();
        self.rms_norm.launch(
            x.access_mut(),
            &unsafe { x_.access_unchecked() },
            &self.model.model_norm.access(),
            self.host.rms_norm_eps(),
            compute,
        );
        // compute.synchronize();
        // println!("model norm:\n{}", map_tensor(&x));

        let mut logits = unsafe {
            logits_dev
                .as_ref()
                .map_physical(|dev| vec![0; dev.access().len()])
        };
        mat_mul(
            &self.cublas,
            &logits_dev.access(),
            0.,
            &x.access(),
            &self.model.lm_head.clone().transpose(&[1, 0]).access(),
            1.,
        );
        compute.synchronize();
        logits_dev
            .access()
            .physical()
            .copy_out(logits.physical_mut());
        // println!("logits:\n{}", logits);

        (requests.into_iter().map(|r| r.id).collect(), logits)
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
            let dev = dev.access();
            let mut buf = vec![0; dev.len()];
            dev.copy_out(&mut buf);
            buf
        })
    }
}
