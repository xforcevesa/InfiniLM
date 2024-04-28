mod kernel;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{safe_tensors::SafeTensors, upos, utok, Blob, FileLoadError};
use gemm::f16;
use itertools::izip;
use kernel::{
    fused_softmax::softmax, gather::gather, mat_mul::mat_mul, rms_norm::rms_norm,
    rotary_embedding::rotary_embedding, swiglu::swiglu,
};
use llama::ConfigJson;
use std::{iter::repeat, path::Path, slice::from_raw_parts};
use tensor::{reslice, slice, split, udim, DataType, LocalSplitable, Tensor};

pub struct Transformer {
    eos_token: utok,
    data_type: DataType,
    nlayers: udim,
    nh: udim,
    nkvh: udim,
    max_seq_len: udim,
    d: udim,
    di: udim,
    epsilon: f32,
    theta: f32,
    safe_tensors: SafeTensors,
}

impl Transformer {
    pub fn embed_tokens(&self) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, "model.embed_tokens.weight")
    }

    pub fn input_layernorm(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "input_layernorm"))
    }

    pub fn w_qkv(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "self_attn.qkv_proj"))
    }

    pub fn w_o(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "self_attn.o_proj"))
    }

    pub fn post_attention_layernorm(&self, layer: udim) -> Tensor<&[u8]> {
        convert(
            &self.safe_tensors,
            layer_name(layer, "post_attention_layernorm"),
        )
    }

    pub fn mlp_gate_up(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "mlp.gate_up_proj"))
    }

    pub fn mlp_down(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "mlp.down_proj"))
    }

    pub fn model_norm(&self) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, "model.norm.weight")
    }

    pub fn lm_head(&self) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, "lm_head.weight")
    }
}

fn layer_name(layer: udim, name: &str) -> String {
    format!("model.layers.{layer}.{name}.weight")
}

fn convert<'a>(tensors: &'a SafeTensors, name: impl AsRef<str>) -> Tensor<&'a [u8]> {
    let tensor = tensors
        .get(name.as_ref())
        .expect(&format!("Tensor {} not found", name.as_ref()));
    let data_type = llama::convert(tensor.dtype);
    let shape = tensor.shape.iter().map(|&x| x as udim).collect::<Vec<_>>();
    Tensor::new(data_type, &shape, tensor.data)
}

impl Model for Transformer {
    type Meta = ();
    type Error = FileLoadError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, _meta: Self::Meta) -> Result<Self, Self::Error> {
        let config = ConfigJson::load(&model_dir)?;
        Ok(Self {
            eos_token: config.eos_token_id,
            data_type: config.torch_dtype,
            nlayers: config.num_hidden_layers as _,
            nh: config.num_attention_heads as _,
            nkvh: config.num_key_value_heads as _,
            max_seq_len: config.max_position_embeddings as _,
            d: config.hidden_size as _,
            di: config.intermediate_size as _,
            epsilon: config.rms_norm_eps,
            theta: config.rope_theta,
            safe_tensors: SafeTensors::load_from_dir(model_dir)?,
        })
    }
}

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn eos_token(&self) -> utok {
        self.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        let dt = self.data_type;
        let nlayers = self.nlayers;
        let nkvh = self.nkvh;
        let max_seq_len = self.max_seq_len;
        let d = self.d;
        let nh = self.nh;
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
        let dt = self.data_type;
        let d = self.d;

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = Tensor::alloc(dt, &[nt, d], Blob::new);
        gather(&mut x, &self.embed_tokens(), tokens);
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

        let dt = self.data_type;
        let d = self.d;
        let nh = self.nh;
        let nkvh = self.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.di;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();

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
        for layer in 0..self.nlayers {
            let (mut x1, qkv) = state!();
            let mut qkv = qkv.slice(&[slice![=>], slice![=> d + dkv + dkv]]);

            let input_layernorm = self.input_layernorm(layer);
            rms_norm(&mut x1, &x, &input_layernorm, self.epsilon);

            let w_qkv = self.w_qkv(layer).transpose(&[1, 0]);
            mat_mul(&mut qkv, 0., &x1, &w_qkv, 1.);

            let (q, k, v) = split!(qkv; [1]: d, dkv, dkv);
            let mut q = q.reshape(&[nt, nh, dh]);
            let mut k = k.reshape(&[nt, nkvh, dh]);
            let v = v.reshape(&[nt, nkvh, dh]);
            let o = x1.reshape(&[nt, nh, dh]);

            rotary_embedding(&mut q, &pos, self.theta);
            rotary_embedding(&mut k, &pos, self.theta);

            let q = q.transpose(&[1, 0, 2]).split(1, &seq_len);
            let k = k.transpose(&[1, 0, 2]).split(1, &seq_len);
            let v = v.transpose(&[1, 0, 2]).split(1, &seq_len);
            let o = o.transpose(&[1, 0, 2]).split(1, &seq_len);

            for (query, q, k, v, mut o) in izip!(&mut queries, q, k, v, o) {
                let pos = query.pos();
                let seq_len = query.seq_len();
                let att_len = query.att_len();
                let Some((mut k_cache, mut v_cache)) = query.cache(layer as _) else {
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
                q.reform_to(&mut q_att);
                k.reform_to(&mut k_cat);
                v.reform_to(&mut v_cat);

                let q_att = q_att.reshape(shape_q1);
                let k_att = k_cache.slice(slice_att).transpose(&[0, 2, 1]);
                let v_att = v_cache.slice(slice_att);

                let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
                mat_mul(&mut att, 0., &q_att, &k_att, head_div);
                let mut att = att.reshape(shape_att1);
                softmax(&mut att);
                let mut x2 = q_att;
                mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1.);

                x2.reshape(shape_q0).reform_to(&mut o);
            }

            let (mut x1, gate_up) = state!();
            let mut gate_up = gate_up.slice(&[slice![=>], slice![=> di + di]]);

            let wo = self.w_o(layer).transpose(&[1, 0]);
            mat_mul(&mut x, 1., &x1, &wo, 1.);

            let post_layernorm = self.post_attention_layernorm(layer);
            rms_norm(&mut x1, &x, &post_layernorm, self.epsilon);

            let w_gate_up = self.mlp_gate_up(layer).transpose(&[1, 0]);
            mat_mul(&mut gate_up, 0., &x1, &w_gate_up, 1.);

            let (mut gate, up) = split!(gate_up; [1]: di, di);
            swiglu(&mut gate, &up);

            let mlp_down = self.mlp_down(layer).transpose(&[1, 0]);
            mat_mul(&mut x, 1., &gate, &mlp_down, 1.);
        }

        x
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.data_type;
        let d = self.d;

        let buf = hidden_state.as_mut_slice();
        let len = d as usize * dt.size();

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

        let lm_head = self.lm_head().transpose(&[1, 0]);
        let mut x = hidden_state.slice(&[slice![begin => dst], slice![=>]]);
        let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        rms_norm(&mut x, &x_, &self.model_norm(), self.epsilon);
        mat_mul(&mut logits, 0., &x, &lm_head, 1.);

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
