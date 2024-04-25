mod kernel;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{safe_tensors::SafeTensorsError, upos, utok, Blob};
use gemm::f16;
use itertools::izip;
use kernel::CpuKernels;
use std::{iter::repeat, path::Path, slice::from_raw_parts};
use tensor::{reslice, slice, split, udim, DataType, LocalSplitable, Tensor};
use transformer::{Kernels, Llama2, Memory};

pub struct Transformer(Memory);

impl Model for Transformer {
    type Meta = ();
    type Error = SafeTensorsError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, _meta: Self::Meta) -> Result<Self, Self::Error> {
        let memory = Memory::load_safetensors(model_dir)?;
        if memory.data_type() == DataType::F16 {
            Ok(Self(memory))
        } else {
            Ok(Self(Memory::cast(&memory, DataType::F16)))
        }
    }
}

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn eos_token(&self) -> utok {
        self.0.eos_token_id()
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

        let mut x = Tensor::alloc(dt, &[nt, d], Blob::new);
        kernels.gather(&mut x, &self.0.embed_tokens(), tokens);
        x
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

#[test]
fn test_infer() {
    use std::time::Instant;

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let t0 = Instant::now();
    let model = <Transformer as Model>::load(model_dir, ()).unwrap();
    let t1 = Instant::now();
    println!("load {:?}", t1 - t0);

    let mut cache = model.new_cache();

    let mut prompt: Vec<utok> = vec![
        29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592, 21106,
        29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
    ];
    let mut pos = 0;

    while prompt != &[model.eos_token()] {
        let token_embedded = CausalLM::token_embed(&model, prompt.iter().copied());

        let queries = [QueryContext {
            cache: Some(&mut cache),
            range: pos..pos + prompt.len() as upos,
        }];
        let hidden_state = CausalLM::forward(&model, queries, token_embedded);

        let decoding = [DecodingMeta {
            num_query: prompt.len(),
            num_decode: 1,
        }];
        let logits = CausalLM::decode(&model, decoding, hidden_state);

        let args = [SampleMeta {
            num_decode: 1,
            args: causal_lm::SampleArgs::default(),
        }];
        let tokens = CausalLM::sample(&model, args, logits);

        println!("{:?}", tokens);
        pos += prompt.len() as upos;
        prompt = tokens;
    }
}
