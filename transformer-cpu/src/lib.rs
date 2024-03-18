mod kernel;
mod storage;

use common::utok;
use gemm::f16;
use kernel::{gather, mat_mul, rms_norm, rotary_embedding, softmax, swiglu};
use storage::Storage;
use tensor::{reslice, slice, udim, DataType, Tensor};

pub type Request<'a, Id> = transformer::Request<'a, Id, Storage>;
pub type LayerCache = transformer::LayerCache<Storage>;
use transformer::{argmax, random};
pub use transformer::{save, Llama2, Memory, SampleArgs};

pub struct Transformer(Box<dyn Llama2>);

impl Transformer {
    #[inline]
    pub fn new(model: Box<dyn Llama2 + 'static>) -> Self {
        Self(model)
    }

    #[inline]
    pub fn new_cache(&self) -> Vec<LayerCache> {
        LayerCache::new_layers(&*self.0, tensor)
    }

    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.0.max_position_embeddings()
    }

    #[inline]
    pub fn eos_token_id(&self) -> utok {
        self.0.eos_token_id()
    }

    pub fn decode<Id>(
        &mut self,
        mut requests: Vec<Request<Id>>,
        sample: &SampleArgs,
    ) -> Vec<(Id, utok)> {
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

        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let di = self.0.intermediate_size() as udim;
        let voc = self.0.vocab_size() as udim;
        let dt = self.0.data_type();
        let epsilon = self.0.rms_norm_eps();
        let theta = self.0.rope_theta();
        let mut pos = Vec::<u32>::with_capacity(nt as usize);
        for request in requests.iter() {
            pos.extend(request.pos..request.att_len());
        }
        let pos = Tensor::new(DataType::U32, &[nt], reslice(&pos));

        let mut x0 = tensor(dt, &[nt, d]);
        let mut x1 = tensor(dt, &[nt, d]);
        let mut qkv = tensor(dt, &[nt, d + dkv + dkv]);
        let mut q_buf = Storage::new((nh * max_seq_len * dh) as usize * dt.size());
        let mut att_buf = Storage::new((nh * max_seq_len * max_att_len) as usize * dt.size());
        let mut gate_up = tensor(dt, &[nt, di + di]);

        let tokens = requests.iter().flat_map(|r| r.tokens).copied();
        gather(&mut x0, &self.0.embed_tokens(), tokens);
        // println!("gather:\n{x0}");

        for layer in 0..self.0.num_hidden_layers() {
            let input_layernorm = self.0.input_layernorm(layer);
            rms_norm(&mut x1, &x0, &input_layernorm, epsilon);
            // println!("layer {layer} input norm:\n{x1}");

            let w_qkv = self.0.w_qkv(layer).transpose(&[1, 0]);
            mat_mul(&mut qkv, 0., &x1, &w_qkv, 1.);
            let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
            let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
            let mut k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
            let mut q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
            // println!("layer {layer} q:\n{q}");
            // println!("layer {layer} k:\n{k}");
            // println!("layer {layer} v:\n{v}");

            rotary_embedding(&mut q, &pos, theta);
            rotary_embedding(&mut k, &pos, theta);
            // println!("layer {layer} rot q:\n{q}");
            // println!("layer {layer} rot k:\n{k}");

            let q = q.as_ref().transpose(&[1, 0, 2]);
            let k = k.as_ref().transpose(&[1, 0, 2]);
            let v = v.as_ref().transpose(&[1, 0, 2]);
            let mut o = x1.as_mut().reshape(&[nt, nh, dh]).transpose(&[1, 0, 2]);

            let q = unsafe { q.map_physical(|u| &**u) };
            let k = unsafe { k.map_physical(|u| &**u) };
            let v = unsafe { v.map_physical(|u| &**u) };

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
                let o = o.as_mut().slice(req_slice);
                let mut o = unsafe { o.map_physical(|u| &mut ***u) };

                let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut q_buf[..]);
                let (k_cache, v_cache) = r.cache[layer].get();
                let k_cat = k_cache.as_mut().slice(cat_slice);
                let v_cat = v_cache.as_mut().slice(cat_slice);
                let mut k_cat = unsafe { k_cat.map_physical(|u| &mut **u) };
                let mut v_cat = unsafe { v_cat.map_physical(|u| &mut **u) };
                q.reform_to(&mut q_att);
                k.reform_to(&mut k_cat);
                v.reform_to(&mut v_cat);

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

                let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
                mat_mul(&mut att, 0., &q_att, &k_att, head_div);
                let mut att = att.reshape(shape_att1);
                softmax(&mut att);
                let mut x2 = q_att;
                mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1.);

                x2.reshape(&[nh, seq_len, dh]).reform_to(&mut o);
                // println!("layer {layer} after attention:\n{}", o);
            }

            let wo = self.0.self_attn_o_proj(layer).transpose(&[1, 0]);
            mat_mul(&mut x0, 1., &x1, &wo, 1.);
            // println!("layer {layer} o_proj:\n{}", x0.access());

            let post_layernorm = self.0.post_attention_layernorm(layer);
            rms_norm(&mut x1, &x0, &post_layernorm, epsilon);
            // println!("layer {layer} post norm:\n{}", x1.access());

            let w_gate_up = self.0.mlp_gate_up(layer).transpose(&[1, 0]);
            mat_mul(&mut gate_up, 0., &x1, &w_gate_up, 1.);
            let mut gate_up = gate_up.split(1, &[di as _, di as _]);
            let up = gate_up.pop().unwrap();
            let mut gate = gate_up.pop().unwrap();
            // println!("layer {layer} gate:\n{}", gate.access());
            // println!("layer {layer} up:\n{}", up.access());

            swiglu(&mut gate, &up);
            // println!("layer {layer} swiglu:\n{}", gate.access());

            let mlp_down = self.0.mlp_down(layer).transpose(&[1, 0]);
            mat_mul(&mut x0, 1., &gate, &mlp_down, 1.);
            // println!("layer {layer} down:\n{}", x0.access());
        }

        let tokens = {
            let (head, others) = requests.split_first().unwrap();
            let begin = head.tokens.len();
            let mut i = begin;
            let mut j = begin;
            let buf = x0.as_mut_slice();
            let len = d as usize * dt.size();
            for r in others {
                i += r.tokens.len();
                j += 1;
                if i > j {
                    buf.copy_within((i - 1) * len..i * len, (j - 1) * len);
                }
            }
            let begin = begin as udim - 1;
            let len = j as udim - begin;
            slice![from begin, take len]
        };

        let mut logits = Tensor::new(
            dt,
            &[tokens.len, voc],
            vec![0u8; (tokens.len * voc) as usize * dt.size()],
        );
        let mut x = x0.slice(&[tokens, slice![all]]);
        // println!("decode slice:\n{}", x.access());

        // 复制一个 x 以实现原地归一化
        let x_ = unsafe {
            x.as_ref()
                .map_physical(|u| std::slice::from_raw_parts(u.as_ptr(), u.len()))
        };
        rms_norm(&mut x, &x_, &self.0.model_norm(), epsilon);
        // println!("model norm:\n{}", x.access());

        let lm_head = self.0.lm_head().transpose(&[1, 0]);
        mat_mul(&mut logits, 0., &x, &lm_head, 1.);
        // println!("logits:\n{}", logits.access());

        let logits: &[f16] = reslice(logits.as_slice());
        requests
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let logits = &kernel::slice!(logits; voc; [i]);
                (
                    r.id,
                    match sample {
                        SampleArgs::Top => argmax(logits),
                        SampleArgs::Random {
                            temperature,
                            top_k,
                            top_p,
                        } => random(logits, *temperature, *top_k, *top_p),
                    },
                )
            })
            .collect()
    }
}

#[inline]
fn tensor(dt: DataType, shape: &[udim]) -> Tensor<Storage> {
    let size = shape.iter().product::<udim>() as usize * dt.size();
    Tensor::new(dt, shape, Storage::new(size))
}

#[test]
fn test_build() {
    use std::{io::ErrorKind::NotFound, time::Instant};
    use transformer::SafeTensorError;

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
