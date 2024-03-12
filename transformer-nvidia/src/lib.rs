#![cfg(detected_cuda)]

mod kernel;
mod parameters;
mod storage;

#[macro_use]
extern crate log;

use ::half::f16;
use common::{upos, utok};
use cublas::Cublas;
use cuda::{CudaDataType::half, DevMem, Stream};
use kernel::{gather, mat_mul, FusedSoftmax, Reform, RmsNormalization, RotaryEmbedding, Swiglu};
use parameters::{LayersParameters, ModelParameters};
use storage::Storage;
use tensor::{slice, udim, DataType, Tensor};

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
    logits_dev: Tensor<Storage<'ctx>>,
    logits: Vec<f16>,
}

impl<'ctx> Transformer<'ctx> {
    pub fn new(host: &'ctx dyn Llama2, preload_layers: usize, stream: &'ctx Stream) -> Self {
        let vocab_size = host.vocab_size();
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
            logits_dev: tensor(host.data_type(), &[1, vocab_size as _], stream),
            logits: vec![f16::ZERO; vocab_size],
            host,
        }
    }

    #[inline]
    pub fn new_cache<'b>(&self, stream: &'b Stream) -> Vec<LayerCache<'b>> {
        LayerCache::new_layers(self.host, |dt, shape| tensor(dt, shape, stream))
    }

    pub fn update(
        &mut self,
        tokens: &[utok],
        cache: &mut [LayerCache],
        pos: upos,
        compute: &Stream<'ctx>,
        transfer: &Stream<'ctx>,
    ) -> Tensor<Storage<'ctx>> {
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
        let pos = transfer.from_host(&(pos..pos + seq_len).collect::<Vec<udim>>());
        let pos = Tensor::new(DataType::U32, &[seq_len], Storage::from(pos));
        // println!("tokens: {tokens:?}");

        let mut x0 = tensor(dt, &[seq_len, d], transfer);
        let e_alloc_x0 = transfer.record();
        let mut x1 = tensor(dt, &[seq_len, d], transfer);
        let qkv = tensor(dt, &[seq_len, d + dkv + dkv], transfer);
        let att = tensor(dt, &[nkvh, head_group * seq_len, att_len], transfer);
        let gate_up = tensor(dt, &[seq_len, di + di], transfer);

        let (mut x2, mut q_att) = if seq_len > 1 {
            (
                // `seq_len x hidden_size` -reshape-> `seq_len x (num_kv_head x head_group x head_dim)` -transpose(1,2,0,3)-> `num_kv_head x head_group x seq_len x head_dim` -reshape-> `num_kv_head x (head_group x seq_len) x head_dim`
                Some(tensor(dt, &[nkvh, head_group * seq_len, dh], transfer)),
                Some(tensor(dt, &[nh, seq_len, dh], transfer)),
            )
        } else {
            (None, None)
        };
        let e_alloc = transfer.record();

        compute.wait_for(&e_alloc_x0);
        gather(
            &mut x0.access_mut(),
            &self.host.embed_tokens(),
            tokens,
            compute,
        );
        // compute.synchronize();
        // println!("gather:\n{}", map_tensor(&x0));

        self.cublas.set_stream(compute);
        compute.wait_for(&e_alloc);
        for (layer, cache) in cache.iter_mut().enumerate() {
            self.layers.load(layer, self.host, transfer);
            let params = self.layers.sync(layer, compute);

            self.rms_norm.launch(
                &mut x1.access_mut(),
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
            let v = qkv.pop().unwrap().reshape(&[seq_len, nkvh, dh]);
            let k = qkv.pop().unwrap().reshape(&[seq_len, nkvh, dh]);
            let q = qkv.pop().unwrap().reshape(&[seq_len, nh, dh]);
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

            let (k_cache, v_cache) = cache.get();
            let k_cat = k_cache.clone().slice(cat_slice);
            let v_cat = v_cache.clone().slice(cat_slice);
            let q_att = if let Some(q_att) = q_att.as_mut() {
                self.reform.launch(&q_att.access(), &q.access(), compute);
                q_att.clone()
            } else {
                q.reshape(&[nh, seq_len, dh])
            };
            self.reform.launch(&k_cat.access(), &k.access(), compute);
            self.reform.launch(&v_cat.access(), &v.access(), compute);

            let q_att = q_att.clone().reshape(&[nkvh, head_group * seq_len, dh]);
            let k_att = k_cache.clone().slice(att_slice);
            let v_att = v_cache.clone().slice(att_slice);
            // compute.synchronize();
            // println!("layer {layer} q attention:\n{}", map_tensor(&q_att));
            // println!("layer {layer} k attention:\n{}", map_tensor(&k_att));
            // println!("layer {layer} v attention:\n{}", map_tensor(&v_att));

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
                {
                    let att = att.clone().reshape(&[nh, seq_len, att_len]);
                    // compute.synchronize();
                    // println!("layer {layer} before softmax:\n{}", map_tensor(&att));
                    self.fused_softmax.launch(&att.access(), compute);
                    // compute.synchronize();
                    // println!("layer {layer} after softmax:\n{}", map_tensor(&att));
                }
                if let Some(x2) = x2.as_mut() {
                    mat_mul(
                        &self.cublas,
                        &x2.access(),
                        0.,
                        &att.access(),
                        &v_att.access(),
                        1.,
                    );
                    self.reform.launch(
                        &x1.clone().reshape(&[seq_len, nh, dh]).access(),
                        &x2.clone()
                            .reshape(&[nh, seq_len, dh])
                            .transpose(&[1, 0, 2])
                            .access(),
                        compute,
                    );
                } else {
                    let x2 = x1.clone().reshape(&[nkvh, head_group * seq_len, dh]);
                    mat_mul(
                        &self.cublas,
                        &x2.access(),
                        0.,
                        &att.access(),
                        &v_att.access(),
                        1.,
                    );
                }
                // compute.synchronize();
                // println!("layer {layer} after attention:\n{}", map_tensor(&x1));
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
                &mut x1.access_mut(),
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
            // compute.synchronize();
            // println!("layer {layer} down:\n{}", map_tensor(&x0));
        }

        x0
    }

    pub fn decode(
        &mut self,
        token: utok,
        cache: &mut [LayerCache],
        pos: upos,
        compute: &Stream<'ctx>,
        transfer: &Stream<'ctx>,
    ) -> &[f16] {
        let mut x = self.update(&[token], cache, pos, compute, transfer);

        compute.wait_for(&self.model.sync_event);
        let src = x.clone();
        self.rms_norm.launch(
            &mut x.access_mut(),
            &unsafe { src.access_unchecked() },
            &self.model.model_norm.access(),
            self.host.rms_norm_eps(),
            compute,
        );
        // compute.synchronize();
        // println!("pos {pos} model norm:\n{}", map_tensor(&x));

        mat_mul(
            &self.cublas,
            &self.logits_dev.access(),
            0.,
            &x.access(),
            &self.model.lm_head.clone().transpose(&[1, 0]).access(),
            1.,
        );
        compute.synchronize();
        self.logits_dev
            .access()
            .physical()
            .copy_out(&mut self.logits);
        // println!("pos {pos} logits:\n{:?}", self.logits);

        &self.logits
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
fn map_tensor(tensor: &Tensor<DevMem>) -> Tensor<Vec<u8>> {
    unsafe {
        tensor.map_physical(|dev| {
            let mut buf = vec![0; dev.len()];
            dev.copy_out(&mut buf);
            buf
        })
    }
}
