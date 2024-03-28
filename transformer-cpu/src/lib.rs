mod kernel;
mod sample;
mod storage;

use common::utok;
use kernel::{gather, mat_mul, rms_norm, rotary_embedding, softmax, swiglu};
use storage::Storage;
use tensor::{reslice, slice, udim, DataType, Tensor};
use transformer::{pos, LayerBuffer, Sample as _};

pub type Request<'a, Id> = transformer::Request<'a, Id, Storage>;
pub type LayerCache = transformer::LayerCache<Storage>;
pub use sample::Sample;
pub use transformer::{save, Llama2, Memory, SampleArgs};

pub struct Transformer(Box<dyn Llama2>);

impl Transformer {
    #[inline]
    pub fn new(model: Box<dyn Llama2 + 'static>) -> Self {
        assert!(model.data_type() == DataType::F16 || model.data_type() == DataType::F32);
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
        &self,
        mut requests: Vec<Request<Id>>,
        sample: &SampleArgs,
    ) -> Vec<(Id, utok)> {
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);
        // 生成词嵌入并预分配空间
        let mut x0 = self.token_embed(&requests);
        let mut x1 = tensor(x0.data_type(), x0.shape());
        let mut buf = LayerBuffer::alloc(&*self.0, &requests, Storage::new);
        // 生成位置张量
        let nt = x0.shape()[0]; // `nt` for number of tokens
        let pos = pos(&requests, nt);
        let pos = Tensor::new(DataType::U32, &[nt], reslice(&pos));
        // 推理
        for layer in 0..self.0.num_hidden_layers() {
            let (q, k, v) = self.before_att(layer, &x0, &mut x1, &mut buf.qkv, &pos);
            let o = &mut x1;
            self.attention(
                layer,
                &mut requests,
                q,
                k,
                v,
                o,
                &mut buf.q_buf,
                &mut buf.att_buf,
            );
            self.after_att(layer, &mut x0, &mut x1, &mut buf.gate_up);
        }
        // 解码
        if requests[0].decode() {
            let x = self.move_decode(&requests, x0);
            let requests = requests.into_iter().map(Request::id).collect();
            Sample.sample(sample, requests, self.logits(x))
        } else {
            vec![]
        }
    }

    fn token_embed<Id>(&self, requests: &[Request<Id>]) -> Tensor<Storage> {
        let dt = self.0.data_type();
        let nt = requests.iter().map(Request::seq_len).sum::<udim>();
        let d = self.0.hidden_size() as udim;
        let mut x0 = tensor(dt, &[nt, d]);

        let tokens = requests.iter().flat_map(Request::tokens).copied();
        gather(&mut x0, &self.0.embed_tokens(), tokens);
        // println!("gather:\n{x0}");

        x0
    }

    fn before_att(
        &self,
        layer: usize,
        x0: &Tensor<Storage>,
        x1: &mut Tensor<Storage>,
        qkv: &mut Tensor<Storage>,
        pos: &Tensor<&[u8]>,
    ) -> (Tensor<Storage>, Tensor<Storage>, Tensor<Storage>) {
        let nt = x0.shape()[0];
        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let epsilon = self.0.rms_norm_eps();
        let theta = self.0.rope_theta();

        let input_layernorm = self.0.input_layernorm(layer);
        rms_norm(x1, x0, &input_layernorm, epsilon);
        // println!("layer {layer} input norm:\n{x1}");

        let w_qkv = self.0.w_qkv(layer).transpose(&[1, 0]);
        mat_mul(qkv, 0., x1, &w_qkv, 1.);
        let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
        let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
        // println!("layer {layer} q:\n{q}");
        // println!("layer {layer} k:\n{k}");
        // println!("layer {layer} v:\n{v}");

        rotary_embedding(&mut q, pos, theta);
        rotary_embedding(&mut k, pos, theta);
        // println!("layer {layer} rot q:\n{q}");
        // println!("layer {layer} rot k:\n{k}");

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
    ) {
        let dt = self.0.data_type();
        let nt = o.shape()[0];
        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let dh = d / nh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();

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

            let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut q_buf[..]);
            let (k_cache, v_cache) = r.cache(layer);
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
            // println!("layer {layer} q attention:\n{q_att}");
            // println!("layer {layer} k attention:\n{k_att}");
            // println!("layer {layer} v attention:\n{v_att}");

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
    }

    fn after_att(
        &self,
        layer: usize,
        x0: &mut Tensor<Storage>,
        x1: &mut Tensor<Storage>,
        gate_up: &mut Tensor<Storage>,
    ) {
        let di = self.0.intermediate_size() as udim;
        let epsilon = self.0.rms_norm_eps();

        let wo = self.0.self_attn_o_proj(layer).transpose(&[1, 0]);
        mat_mul(x0, 1., x1, &wo, 1.);
        // println!("layer {layer} o_proj:\n{x0}");

        let post_layernorm = self.0.post_attention_layernorm(layer);
        rms_norm(x1, x0, &post_layernorm, epsilon);
        // println!("layer {layer} post norm:\n{x1}");

        let w_gate_up = self.0.mlp_gate_up(layer).transpose(&[1, 0]);
        mat_mul(gate_up, 0., x1, &w_gate_up, 1.);
        let mut gate_up = gate_up.split(1, &[di as _, di as _]);
        let up = gate_up.pop().unwrap();
        let mut gate = gate_up.pop().unwrap();
        // println!("layer {layer} gate:\n{gate}");
        // println!("layer {layer} up:\n{up}");

        swiglu(&mut gate, &up);
        // println!("layer {layer} swiglu:\n{gate}");

        let mlp_down = self.0.mlp_down(layer).transpose(&[1, 0]);
        mat_mul(x0, 1., &gate, &mlp_down, 1.);
        // println!("layer {layer} down:\n{x0}");
    }

    fn move_decode<Id>(
        &self,
        requests: &[Request<Id>],
        mut x0: Tensor<Storage>,
    ) -> Tensor<Storage> {
        let buf = x0.as_mut_slice();
        let len = self.0.hidden_size() * self.0.data_type().size();

        let (head, others) = requests.split_first().unwrap();
        let begin = head.seq_len() as usize - 1;

        let mut src = begin;
        let mut dst = begin;
        for r in others {
            src += r.seq_len() as usize;
            if r.decode() {
                dst += 1;
                if dst < src {
                    buf.copy_within(src * len..(src + 1) * len, dst * len);
                }
            }
        }

        x0.slice(&[slice![from begin, until dst + 1], slice![all]])
    }

    fn logits(&self, mut x: Tensor<Storage>) -> Tensor<Storage> {
        let dt = self.0.data_type();
        let voc = self.0.vocab_size() as udim;
        let epsilon = self.0.rms_norm_eps();

        let mut logits = tensor(dt, &[x.shape()[0], voc]);
        // println!("decode slice:\n{x}");

        // 复制一个 x 以实现原地归一化
        let x_ = unsafe {
            x.as_ref()
                .map_physical(|u| std::slice::from_raw_parts(u.as_ptr(), u.len()))
        };
        rms_norm(&mut x, &x_, &self.0.model_norm(), epsilon);
        // println!("model norm:\n{x}");

        let lm_head = self.0.lm_head().transpose(&[1, 0]);
        mat_mul(&mut logits, 0., &x, &lm_head, 1.);
        // println!("logits:\n{logits}");

        logits
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
