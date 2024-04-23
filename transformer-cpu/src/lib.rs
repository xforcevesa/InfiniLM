mod kernel;

use causal_lm::{CausalLM, DecodingMeta, QueryContext, SampleMeta};
use common::{upos, utok, Blob};
use gemm::f16;
use itertools::izip;
use kernel::CpuKernels;
use std::{iter::repeat, path::Path, slice::from_raw_parts};
use tensor::{reslice, slice, split, udim, DataType, LocalSplitable, Tensor};
use transformer::{pos, Kernels, LayerBuffer, LayerCache, Llama2, Memory, Request, SampleArgs};

pub struct Transformer(Memory);

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn eos_token(&self) -> utok {
        self.0.eos_token_id()
    }

    #[inline]
    fn load(model_dir: impl AsRef<Path>) -> Self {
        let memory = Memory::load_safetensors(model_dir).unwrap();
        if memory.data_type() == DataType::F16 {
            Self(memory)
        } else {
            Self(Memory::cast(&memory, DataType::F16))
        }
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        let dt = self.0.data_type();
        let nlayers = self.0.num_hidden_layers() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let max_seq_len = self.0.max_position_embeddings() as udim;
        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;

        Tensor::alloc(dt, &[nlayers, 2, nkvh, max_seq_len, d / nh], Blob::new)
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        let &[_nlayers, 2, _nkvh, max_seq_len, _dh] = cache.shape() else {
            panic!()
        };
        assert!(pos <= max_seq_len);
        let slice = [
            slice![=>],
            slice![=>],
            slice![=>],
            slice![=>pos],
            slice![=>],
        ];

        let mut ans = Tensor::alloc(cache.data_type(), cache.shape(), Blob::new);
        cache
            .as_ref()
            .slice(&slice)
            .map_physical(|u| &**u)
            .reform_to(&mut ans.as_mut().slice(&slice).map_physical(|u| &mut **u));
        ans
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let dt = self.0.data_type();
        let d = self.0.hidden_size() as udim;
        let kernels = CpuKernels::new(&self.0);

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x0 = Tensor::alloc(dt, &[nt, d], Blob::new);
        kernels.gather(&mut x0, &self.0.embed_tokens(), tokens);
        x0
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        let mut queries = queries.into_iter().collect::<Vec<_>>();
        let mut nt = 0;
        let mut max_seq_len = 0;
        let mut max_att_len = 0;
        let seq_len = queries
            .iter()
            .map(|q| {
                let seq = q.seq_len();
                let att = q.att_len();
                nt += seq;
                max_seq_len = max_seq_len.max(seq);
                max_att_len = max_att_len.max(att);
                seq
            })
            .collect::<Vec<_>>();

        let dt = self.0.data_type();
        let d = self.0.hidden_size() as udim;
        let nh = self.0.num_attention_heads() as udim;
        let nkvh = self.0.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.0.intermediate_size() as udim;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let kernels = CpuKernels::new(&self.0);

        let reusing = (d + dkv + dkv).max(di + di);
        let mut state_buf = Tensor::alloc(dt, &[nt, d + reusing], Blob::new);
        macro_rules! state {
            () => {
                split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing)
            };
        }

        let mut q_buf = Blob::new((nh * max_seq_len * dh) as usize * dt.size());
        let mut att_buf = Blob::new((nh * max_seq_len * max_att_len) as usize * dt.size());
        let pos = causal_lm::pos(&queries, nt);
        let pos = pos.as_ref().map_physical(|u| reslice(u));

        let mut x = token_embedded;
        for layer in 0..self.0.num_hidden_layers() {
            let (mut x1, qkv) = state!();
            let mut qkv = qkv.slice(&[slice![=>], slice![=> d + dkv + dkv]]);

            let input_layernorm = self.0.input_layernorm(layer);
            kernels.rms_norm(&mut x1, &x, &input_layernorm);

            let w_qkv = self.0.w_qkv(layer).transpose(&[1, 0]);
            kernels.mat_mul(&mut qkv, 0., &x1, &w_qkv, 1.);

            let (q, k, v) = split!(qkv; [1]: d, dkv, dkv);
            let mut q = q.reshape(&[nt, nh, dh]);
            let mut k = k.reshape(&[nt, nkvh, dh]);
            let v = v.reshape(&[nt, nkvh, dh]);
            let o = x1.reshape(&[nt, nh, dh]);

            kernels.rotary_embedding(&mut q, &pos);
            kernels.rotary_embedding(&mut k, &pos);

            let q = q.transpose(&[1, 0, 2]).split(1, &seq_len);
            let k = k.transpose(&[1, 0, 2]).split(1, &seq_len);
            let v = v.transpose(&[1, 0, 2]).split(1, &seq_len);
            let o = o.transpose(&[1, 0, 2]).split(1, &seq_len);

            for (query, q, k, v, mut o) in izip!(&mut queries, q, k, v, o) {
                let pos = query.pos();
                let seq_len = query.seq_len();
                let att_len = query.att_len();
                let Some((mut k_cache, mut v_cache)) = query.cache(layer) else {
                    continue;
                };

                let slice_cat = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
                let slice_att = &[slice![=>], slice![      => att_len], slice![=>]];
                let shape_q0 = &[nkvh * head_group, seq_len, dh];
                let shape_q1 = &[nkvh, head_group * seq_len, dh];
                let shape_att0 = &[nkvh, head_group * seq_len, att_len];
                let shape_att1 = &[nkvh * head_group, seq_len, att_len];

                let mut q_att = Tensor::new(dt, shape_q0, &mut q_buf[..]);
                let mut k_cat = k_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                let mut v_cat = v_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                kernels.reform(&mut q_att, &q);
                kernels.reform(&mut k_cat, &k);
                kernels.reform(&mut v_cat, &v);

                let q_att = q_att.reshape(shape_q1);
                let k_att = k_cache.slice(slice_att).transpose(&[0, 2, 1]);
                let v_att = v_cache.slice(slice_att);

                let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
                kernels.mat_mul(&mut att, 0., &q_att, &k_att, head_div);
                let mut att = att.reshape(shape_att1);
                kernels.softmax(&mut att);
                let mut x2 = q_att;
                kernels.mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1.);

                kernels.reform(&mut o, &x2.reshape(shape_q0));
            }

            let (mut x1, gate_up) = state!();
            let mut gate_up = gate_up.slice(&[slice![=>], slice![=> di + di]]);

            let wo = self.0.self_attn_o_proj(layer).transpose(&[1, 0]);
            kernels.mat_mul(&mut x, 1., &x1, &wo, 1.);

            let post_layernorm = self.0.post_attention_layernorm(layer);
            kernels.rms_norm(&mut x1, &x, &post_layernorm);

            let w_gate_up = self.0.mlp_gate_up(layer).transpose(&[1, 0]);
            kernels.mat_mul(&mut gate_up, 0., &x1, &w_gate_up, 1.);

            let (mut gate, up) = split!(gate_up; [1]: di, di);
            kernels.swiglu(&mut gate, &up);

            let mlp_down = self.0.mlp_down(layer).transpose(&[1, 0]);
            kernels.mat_mul(&mut x, 1., &gate, &mlp_down, 1.);
        }

        x
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.0.data_type();
        let d = self.0.hidden_size();
        let voc = self.0.vocab_size() as udim;
        let kernels = CpuKernels::new(&self.0);

        let buf = hidden_state.as_mut_slice();
        let len = d * dt.size();

        let mut iter = decoding.into_iter();
        let mut begin = 0;
        let mut src = 0;
        let mut dst = 0;
        for DecodingMeta {
            num_query,
            num_decode,
        } in iter.by_ref()
        {
            begin += num_query;
            if num_decode > 0 {
                src = begin;
                dst = begin;
                begin -= num_decode;
                break;
            }
        }
        for DecodingMeta {
            num_query,
            num_decode,
        } in iter
        {
            src += num_query - num_decode;
            if src > dst {
                for _ in 0..num_decode {
                    buf.copy_within(src * len..(src + 1) * len, dst * len);
                    src += 1;
                    dst += 1;
                }
            } else {
                src += num_decode;
                dst += num_decode;
            }
        }

        if dst == begin {
            return Tensor::alloc(dt, &[0, d as _], Blob::new);
        }

        let mut x = hidden_state.slice(&[slice![begin => dst], slice![=>]]);
        let mut logits = Tensor::alloc(dt, &[x.shape()[0], voc], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        kernels.rms_norm(&mut x, &x_, &self.0.model_norm());

        let lm_head = self.0.lm_head().transpose(&[1, 0]);
        kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);

        logits
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let &[_, voc] = logits.shape() else { panic!() };
        let logits: &[f16] = reslice(logits.as_slice());
        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| args.random(&kernel::slice!(logits; voc; [i])))
            .collect()
    }
}

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
        LayerCache::new_layers(&self.0, |dt, shape| Tensor::alloc(dt, shape, Blob::new))
    }

    fn decode<Id>(
        &self,
        mut requests: Vec<Request<Id, Self::Cache>>,
    ) -> (Vec<Id>, Tensor<Self::Cache>) {
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);
        // 生成词嵌入并预分配空间
        let mut x0 = self.token_embed(&requests);
        let mut x1 = Tensor::alloc(x0.data_type(), x0.shape(), Blob::new);
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

        let mut x0 = Tensor::alloc(dt, &[nt, d], Blob::new);
        let tokens = requests.iter().flat_map(Request::tokens).copied();
        kernels.gather(&mut x0, &self.0.embed_tokens(), tokens);

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

        let w_qkv = self.0.w_qkv(layer).transpose(&[1, 0]);
        kernels.mat_mul(qkv, 0., x1, &w_qkv, 1.);

        let (q, k, v) = split!(qkv; [1]: d, dkv, dkv);
        let mut q = q.reshape(&[nt, nh, dh]);
        let mut k = k.reshape(&[nt, nkvh, dh]);
        let v = v.reshape(&[nt, nkvh, dh]);

        kernels.rotary_embedding(&mut q, pos);
        kernels.rotary_embedding(&mut k, pos);

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

        let q = q.as_ref().transpose(&[1, 0, 2]).map_physical(|u| &**u);
        let k = k.as_ref().transpose(&[1, 0, 2]).map_physical(|u| &**u);
        let v = v.as_ref().transpose(&[1, 0, 2]).map_physical(|u| &**u);
        let mut o = o.as_mut().reshape(&[nt, nh, dh]).transpose(&[1, 0, 2]);

        let mut req = 0;
        for r in requests.iter_mut() {
            let pos = r.pos();
            let seq_len = r.seq_len();
            let att_len = r.att_len();

            let req_slice = &[slice![=>], slice![req =>=> seq_len], slice![=>]];
            let cat_slice = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
            let att_slice = &[slice![=>], slice![      => att_len], slice![=>]];
            req += seq_len;

            let q = q.clone().slice(req_slice);
            let k = k.clone().slice(req_slice);
            let v = v.clone().slice(req_slice);
            let mut o = o.as_mut().slice(req_slice).map_physical(|u| &mut ***u);

            let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut q_buf[..]);
            let (k_cache, v_cache) = r.cache(layer);
            let mut k_cat = k_cache.as_mut().slice(cat_slice).map_physical(|u| &mut **u);
            let mut v_cat = v_cache.as_mut().slice(cat_slice).map_physical(|u| &mut **u);
            kernels.reform(&mut q_att, &q);
            kernels.reform(&mut k_cat, &k);
            kernels.reform(&mut v_cat, &v);

            let q_att = q_att.reshape(&[nkvh, head_group * seq_len, dh]);
            let k_att = k_cache
                .as_ref()
                .slice(att_slice)
                .transpose(&[0, 2, 1])
                .map_physical(|u| &**u);
            let v_att = v_cache.as_ref().slice(att_slice).map_physical(|u| &**u);

            let shape_att0 = &[nkvh, head_group * seq_len, att_len];
            let shape_att1 = &[nkvh * head_group, seq_len, att_len];

            let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
            kernels.mat_mul(&mut att, 0., &q_att, &k_att, head_div);
            let mut att = att.reshape(shape_att1);
            kernels.softmax(&mut att);
            let mut x2 = q_att;
            kernels.mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1.);

            kernels.reform(&mut o, &x2.reshape(&[nh, seq_len, dh]));
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

        let post_layernorm = self.0.post_attention_layernorm(layer);
        kernels.rms_norm(x1, x0, &post_layernorm);

        let w_gate_up = self.0.mlp_gate_up(layer).transpose(&[1, 0]);
        kernels.mat_mul(gate_up, 0., x1, &w_gate_up, 1.);

        let (mut gate, up) = split!(gate_up; [1]: di, di);
        kernels.swiglu(&mut gate, &up);

        let mlp_down = self.0.mlp_down(layer).transpose(&[1, 0]);
        kernels.mat_mul(x0, 1., &gate, &mlp_down, 1.);
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

        x0.slice(&[slice![begin => dst + 1], slice![=>]])
    }

    fn logits(&self, mut x: Tensor<Blob>) -> Tensor<Blob> {
        let dt = self.0.data_type();
        let voc = self.0.vocab_size() as udim;
        let kernels = CpuKernels::new(&self.0);

        let mut logits = Tensor::alloc(dt, &[x.shape()[0], voc], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = unsafe {
            x.as_ref()
                .map_physical(|u| from_raw_parts(u.as_ptr(), u.len()))
        };
        kernels.rms_norm(&mut x, &x_, &self.0.model_norm());

        let lm_head = self.0.lm_head().transpose(&[1, 0]);
        kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);

        logits
    }
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
    let safetensors = Memory::load_safetensors(model_dir);
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
