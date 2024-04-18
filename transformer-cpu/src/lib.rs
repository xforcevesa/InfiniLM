mod kernel;

use common::{utok, Blob};
use gemm::f16;
use kernel::CpuKernels;
use tensor::{reslice, slice, udim, DataType, LocalSplitable, Tensor};
use transformer::{pos, Kernels, LayerBuffer, LayerCache, Llama2, Memory, Request, SampleArgs};

pub struct Transformer(Memory);

impl transformer::Transformer for Transformer {
    type Cache = Blob;

    #[inline]
    fn max_position_embeddings(&self) -> usize {
        self.0.max_position_embeddings()
    }

    #[inline]
    fn eos_token(&self) -> utok {
        self.0.eos_token_id()
    }

    #[inline]
    fn new_cache(&self) -> Vec<LayerCache<Self::Cache>> {
        LayerCache::new_layers(&self.0, tensor)
    }

    fn decode<Id>(
        &self,
        mut requests: Vec<Request<Id, Self::Cache>>,
    ) -> (Vec<Id>, Tensor<Self::Cache>) {
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);
        // 生成词嵌入并预分配空间
        let mut x0 = self.token_embed(&requests);
        let mut x1 = tensor(x0.data_type(), x0.shape());
        let mut buf = LayerBuffer::alloc(&self.0, &requests, Blob::new);
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
            (requests, self.logits(x))
        } else {
            todo!()
        }
    }

    fn sample<Id>(
        &self,
        args: &SampleArgs,
        requests: Vec<Id>,
        logits: Tensor<Self::Cache>,
    ) -> Vec<(Id, utok)> {
        let &[_, voc] = logits.shape() else { panic!() };
        let dt = logits.data_type();

        macro_rules! sample {
                ($ty:ty) => {{
                    let logits: &[$ty] = reslice(logits.as_slice());
                    requests
                        .into_iter()
                        .enumerate()
                        .map(|(i, id)| (id, args.random(&kernel::slice!(logits; voc; [i]))))
                        .collect()
                }};
            }

        match dt {
            DataType::F16 => sample!(f16),
            DataType::F32 => sample!(f32),
            _ => unreachable!(),
        }
    }
}

type Splitable = LocalSplitable<Blob>;

impl Transformer {
    #[inline]
    pub fn new(model: Memory) -> Self {
        assert!(model.data_type() == DataType::F16 || model.data_type() == DataType::F32);
        Self(model)
    }

    fn token_embed<Id>(&self, requests: &[Request<Id, Blob>]) -> Tensor<Blob> {
        let dt = self.0.data_type();
        let nt = requests.iter().map(Request::seq_len).sum::<udim>();
        let d = self.0.hidden_size() as udim;
        let kernels = CpuKernels::new(&self.0);

        let mut x0 = tensor(dt, &[nt, d]);
        let tokens = requests.iter().flat_map(Request::tokens).copied();
        kernels.gather(&mut x0, &self.0.embed_tokens(), tokens);
        // println!("gather:\n{x0}");

        x0
    }

    fn before_att(
        &self,
        layer: usize,
        x0: &Tensor<Blob>,
        x1: &mut Tensor<Blob>,
        qkv: &mut Tensor<Splitable>,
        pos: &Tensor<&[u8]>,
    ) -> (Tensor<Splitable>, Tensor<Splitable>, Tensor<Splitable>) {
        let nt = x0.shape()[0];
        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let kernels = CpuKernels::new(&self.0);

        let input_layernorm = self.0.input_layernorm(layer);
        kernels.rms_norm(x1, x0, &input_layernorm);
        // println!("layer {layer} input norm:\n{x1}");

        let w_qkv = self.0.w_qkv(layer).transpose(&[1, 0]);
        kernels.mat_mul(qkv, 0., x1, &w_qkv, 1.);
        let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
        let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
        // println!("layer {layer} q:\n{q}");
        // println!("layer {layer} k:\n{k}");
        // println!("layer {layer} v:\n{v}");

        kernels.rotary_embedding(&mut q, pos);
        kernels.rotary_embedding(&mut k, pos);
        // println!("layer {layer} rot q:\n{q}");
        // println!("layer {layer} rot k:\n{k}");

        (q, k, v)
    }

    fn attention<Id>(
        &self,
        layer: usize,
        requests: &mut [Request<Id, Blob>],
        q: Tensor<Splitable>,
        k: Tensor<Splitable>,
        v: Tensor<Splitable>,
        o: &mut Tensor<Blob>,
        q_buf: &mut Blob,
        att_buf: &mut Blob,
    ) {
        let dt = self.0.data_type();
        let nt = o.shape()[0];
        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let dh = d / nh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let kernels = CpuKernels::new(&self.0);

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
            kernels.reform(&mut q_att, &q);
            kernels.reform(&mut k_cat, &k);
            kernels.reform(&mut v_cat, &v);

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
            kernels.mat_mul(&mut att, 0., &q_att, &k_att, head_div);
            let mut att = att.reshape(shape_att1);
            kernels.softmax(&mut att);
            let mut x2 = q_att;
            kernels.mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1.);

            kernels.reform(&mut o, &x2.reshape(&[nh, seq_len, dh]));
            // println!("layer {layer} after attention:\n{}", o);
        }
    }

    fn after_att(
        &self,
        layer: usize,
        x0: &mut Tensor<Blob>,
        x1: &mut Tensor<Blob>,
        gate_up: &mut Tensor<Splitable>,
    ) {
        let di = self.0.intermediate_size() as udim;
        let kernels = CpuKernels::new(&self.0);

        let wo = self.0.self_attn_o_proj(layer).transpose(&[1, 0]);
        kernels.mat_mul(x0, 1., x1, &wo, 1.);
        // println!("layer {layer} o_proj:\n{x0}");

        let post_layernorm = self.0.post_attention_layernorm(layer);
        kernels.rms_norm(x1, x0, &post_layernorm);
        // println!("layer {layer} post norm:\n{x1}");

        let w_gate_up = self.0.mlp_gate_up(layer).transpose(&[1, 0]);
        kernels.mat_mul(gate_up, 0., x1, &w_gate_up, 1.);
        let mut gate_up = gate_up.split(1, &[di as _, di as _]);
        let up = gate_up.pop().unwrap();
        let mut gate = gate_up.pop().unwrap();
        // println!("layer {layer} gate:\n{gate}");
        // println!("layer {layer} up:\n{up}");

        kernels.swiglu(&mut gate, &up);
        // println!("layer {layer} swiglu:\n{gate}");

        let mlp_down = self.0.mlp_down(layer).transpose(&[1, 0]);
        kernels.mat_mul(x0, 1., &gate, &mlp_down, 1.);
        // println!("layer {layer} down:\n{x0}");
    }

    fn move_decode<Id>(
        &self,
        requests: &[Request<Id, Blob>],
        mut x0: Tensor<Blob>,
    ) -> Tensor<Blob> {
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

    fn logits(&self, mut x: Tensor<Blob>) -> Tensor<Blob> {
        let dt = self.0.data_type();
        let voc = self.0.vocab_size() as udim;
        let kernels = CpuKernels::new(&self.0);

        let mut logits = tensor(dt, &[x.shape()[0], voc]);
        // println!("decode slice:\n{x}");

        // 复制一个 x 以实现原地归一化
        let x_ = unsafe {
            x.as_ref()
                .map_physical(|u| std::slice::from_raw_parts(u.as_ptr(), u.len()))
        };
        kernels.rms_norm(&mut x, &x_, &self.0.model_norm());
        // println!("model norm:\n{x}");

        let lm_head = self.0.lm_head().transpose(&[1, 0]);
        kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);
        // println!("logits:\n{logits}");

        logits
    }
}

#[inline]
fn tensor(dt: DataType, shape: &[udim]) -> Tensor<Blob> {
    Tensor::alloc(dt, shape, Blob::new)
}

#[test]
fn test_build() {
    use common::safe_tensors::SafeTensorsError;
    use std::{io::ErrorKind::NotFound, time::Instant};
    use transformer::Memory;

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let t0 = Instant::now();
    let safetensors = Memory::load_safetensors_from_dir(model_dir);
    let t1 = Instant::now();
    println!("mmap {:?}", t1 - t0);

    let safetensors = match safetensors {
        Ok(m) => m,
        Err(SafeTensorsError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let t0 = Instant::now();
    let _transformer = Transformer::new(safetensors);
    let t1 = Instant::now();
    println!("build transformer {:?}", t1 - t0);
}
