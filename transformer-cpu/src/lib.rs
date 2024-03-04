mod cache;
mod kernel;
mod storage;

use common::{upos, utok};
use gemm::f16;
use kernel::{gather, mat_mul, rms_norm, rms_norm_inplace, rotary_embedding, softmax, swiglu};
use model_parameters::{Llama2, Memory};
use storage::Storage;
use tensor::{reslice, reslice_mut, slice, udim, DataType, Tensor};

pub use cache::LayerCache;
pub extern crate model_parameters;

pub struct Transformer {
    model: Box<dyn Llama2>,
    logits: Vec<f32>,
}

impl Transformer {
    #[inline]
    pub fn new(model: Box<dyn Llama2>) -> Self {
        Self {
            logits: vec![0.; model.vocab_size()],
            model: match model.data_type() {
                DataType::BF16 | DataType::F32 => Box::new(Memory::cast(&*model, DataType::F16)),
                _ => model,
            },
        }
    }

    #[inline]
    pub fn new_cache(&self) -> Vec<LayerCache> {
        LayerCache::new_layers(&*self.model)
    }

    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.model.max_position_embeddings()
    }

    pub fn update(
        &self,
        tokens: &[&[utok]],
        cache: &mut [LayerCache],
        pos: upos,
    ) -> Tensor<Storage> {
        let seq_len = tokens.iter().map(|s| s.len()).sum::<usize>() as udim;
        let d = self.model.hidden_size() as udim;
        let nh = self.model.num_attention_heads() as udim;
        let nkvh = self.model.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let di = self.model.intermediate_size() as udim;
        let dt = self.model.data_type();
        let epsilon = self.model.rms_norm_eps();
        let theta = self.model.rope_theta();
        let att_len = pos + seq_len;
        let cat_slice = &[slice![all], slice![pos; 1; seq_len], slice![all]];
        let att_slice = &[slice![all], slice![  0; 1; att_len], slice![all]];
        let pos = (pos..pos + seq_len).collect::<Vec<udim>>();
        let pos = Tensor::new(DataType::U32, &[seq_len], reslice::<udim, u8>(&pos));
        // println!("tokens: {tokens:?}");

        let mut x0 = tensor(dt, &[seq_len, d]);
        let mut x1 = tensor(dt, &[seq_len, d]);
        let mut qkv = tensor(dt, &[seq_len, d + dkv + dkv]);
        let mut att = tensor(dt, &[nkvh, head_group * seq_len, att_len]);
        let mut gate_up = tensor(dt, &[seq_len, di + di]);

        let (mut x2, mut q_att) = if seq_len > 1 {
            (
                // `seq_len x hidden_size` -reshape-> `seq_len x (num_kv_head x head_group x head_dim)` -transpose(1,2,0,3)-> `num_kv_head x head_group x seq_len x head_dim` -reshape-> `num_kv_head x (head_group x seq_len) x head_dim`
                Some(tensor(dt, &[nkvh, head_group * seq_len, dh])),
                Some(tensor(dt, &[nh, seq_len, dh])),
            )
        } else {
            (None, None)
        };

        gather(x0.access_mut(), &self.model.embed_tokens(), tokens);
        // println!("gather:\n{}", x0.access());

        for (layer, cache) in cache.iter_mut().enumerate() {
            let input_layernorm = self.model.input_layernorm(layer);
            rms_norm(x1.access_mut(), &x0.access(), &input_layernorm, epsilon);
            // println!("layer {layer} input norm:\n{}", x1.access());
            let w_qkv = self.model.w_qkv(layer).transpose(&[1, 0]);
            mat_mul(qkv.access_mut(), 0., &x1.access_mut(), &w_qkv, 1.);
            let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
            let v = qkv.pop().unwrap().reshape(&[seq_len, nkvh, dh]);
            let mut k = qkv.pop().unwrap().reshape(&[seq_len, nkvh, dh]);
            let mut q = qkv.pop().unwrap().reshape(&[seq_len, nh, dh]);
            // println!("layer {layer} q:\n{}", q.access());
            // println!("layer {layer} k:\n{}", k.access());
            // println!("layer {layer} v:\n{}", v.access());
            rotary_embedding(q.access_mut(), &pos, theta);
            rotary_embedding(k.access_mut(), &pos, theta);
            // println!("layer {layer} rot q:\n{}", q.access());
            // println!("layer {layer} rot k:\n{}", k.access());
            let q = q.transpose(&[1, 0, 2]);
            let k = k.transpose(&[1, 0, 2]);
            let v = v.transpose(&[1, 0, 2]);

            let (k_cache, v_cache) = cache.get();
            let mut k_cat = k_cache.clone().slice(cat_slice);
            let mut v_cat = v_cache.clone().slice(cat_slice);
            let q_att = if let Some(q_att) = q_att.as_mut() {
                q.access().reform_to(&mut q_att.access_mut());
                q_att.clone()
            } else {
                q.reshape(&[nh, seq_len, dh])
            };
            k.access().reform_to(&mut k_cat.access_mut());
            v.access().reform_to(&mut v_cat.access_mut());

            let q_att = q_att.clone().reshape(&[nkvh, head_group * seq_len, dh]);
            let k_att = k_cache.clone().slice(att_slice);
            let v_att = v_cache.clone().slice(att_slice);
            // println!("layer {layer} q attention:\n{}", q_att.access());
            // println!("layer {layer} k attention:\n{}", k_att.access());
            // println!("layer {layer} v attention:\n{}", v_att.access());

            {
                let k_att = k_att.transpose(&[0, 2, 1]);
                mat_mul(
                    att.access_mut(),
                    0.,
                    &q_att.access(),
                    &k_att.access(),
                    head_div,
                );
                {
                    let mut att = att.clone().reshape(&[nh, seq_len, att_len]);
                    // println!("layer {layer} before softmax:\n{}", att.access());
                    softmax(att.access_mut());
                    // println!("layer {layer} after softmax:\n{}", att.access());
                }
                if let Some(x2) = x2.as_mut() {
                    mat_mul(x2.access_mut(), 0., &att.access(), &v_att.access(), 1.);
                    let x2 = x2.clone().reshape(&[nh, seq_len, dh]).transpose(&[1, 0, 2]);
                    let mut x1 = x1.clone().reshape(&[seq_len, nh, dh]);
                    x2.access().reform_to(&mut x1.access_mut());
                } else {
                    let mut x2 = x1.clone().reshape(&[nkvh, head_group * seq_len, dh]);
                    mat_mul(x2.access_mut(), 0., &att.access(), &v_att.access(), 1.);
                }
                // println!("layer {layer} after attention:\n{}", x1.access());
            }

            let wo = self.model.self_attn_o_proj(layer).transpose(&[1, 0]);
            mat_mul(x0.access_mut(), 1., &x1.access(), &wo, 1.);
            // println!("layer {layer} o_proj:\n{}", x0.access());

            let post_layernorm = self.model.post_attention_layernorm(layer);
            rms_norm(x1.access_mut(), &x0.access(), &post_layernorm, epsilon);
            // println!("layer {layer} post norm:\n{}", x1.access());

            let w_gate_up = self.model.mlp_gate_up(layer).transpose(&[1, 0]);
            mat_mul(gate_up.access_mut(), 0., &x1.access(), &w_gate_up, 1.);
            let mut gate_up = gate_up.split(1, &[di as _, di as _]);
            let up = gate_up.pop().unwrap();
            let mut gate = gate_up.pop().unwrap();
            // println!("layer {layer} gate:\n{}", gate.access());
            // println!("layer {layer} up:\n{}", up.access());

            swiglu(gate.access_mut(), unsafe { &up.access_unchecked() });
            // println!("layer {layer} swiglu:\n{}", gate.access());

            let mlp_down = self.model.mlp_down(layer).transpose(&[1, 0]);
            mat_mul(x0.access_mut(), 1., &gate.access(), &mlp_down, 1.);
            // println!("layer {layer} down:\n{}", x0.access());
        }

        x0
    }

    pub fn forward(&mut self, token: utok, cache: &mut [LayerCache], pos: upos) -> &[f32] {
        let mut x = self.update(&[&[token]], cache, pos);

        let model_norm = self.model.model_norm();
        rms_norm_inplace(&mut x.access_mut(), &model_norm, self.model.rms_norm_eps());
        // println!("pos {pos} model norm:\n{}", x.access());

        let dt = self.model.data_type();
        let voc = self.model.vocab_size() as udim;
        mat_mul(
            Tensor::new(dt, &[1, voc], reslice_mut(&mut self.logits)),
            0.,
            &x.access(),
            &self.model.lm_head().transpose(&[1, 0]),
            1.,
        );
        // println!("pos {pos} logits:\n{}", logits);

        match self.model.data_type() {
            DataType::F32 => {}
            DataType::F16 => {
                let ptr = self.logits.as_ptr().cast::<f16>();
                let len = self.model.vocab_size();
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
fn tensor(dt: DataType, shape: &[udim]) -> Tensor<Storage> {
    Tensor::new(
        dt,
        shape,
        Storage::new(shape.iter().product::<udim>() as usize * dt.size()),
    )
}

#[test]
fn test_build() {
    use model_parameters::SafeTensorError;
    use std::{io::ErrorKind::NotFound, time::Instant};

    let t0 = Instant::now();
    let safetensors = Memory::load_safetensors_from_dir("../../TinyLlama-1.1B-Chat-v1.0");
    let t1 = Instant::now();
    println!("mmap {:?}", t1 - t0);

    let safetensors = match safetensors {
        Ok(m) => m,
        Err(SafeTensorError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let t0 = Instant::now();
    let _transformer = Transformer::new(Box::new(safetensors));
    let t1 = Instant::now();
    println!("build transformer {:?}", t1 - t0);
}
