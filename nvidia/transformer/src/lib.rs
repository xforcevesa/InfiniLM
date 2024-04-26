#![cfg(detected_cuda)]

mod parameters;

#[macro_use]
extern crate log;

use ::half::f16;
use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common_nv::{
    cuda::{memcpy_d2h, DevMemSpore},
    slice, split, udim, upos, utok, DataType, LocalSplitable, NvidiaKernels, NvidiaKernelsPtx,
    SafeTensorsError, Tensor,
};
use cuda::{Context, ContextResource, ContextSpore, Device, StreamSpore};
use itertools::izip;
use parameters::{LayersParameters, ModelParameters};
use std::{
    iter::repeat,
    path::Path,
    slice::from_raw_parts,
    sync::{Arc, Mutex},
    time::Instant,
};
use transformer::{Kernels, Llama2, Memory};

pub use common_nv::{cuda, synchronize};

pub struct Transformer {
    host: Memory,
    model: ModelParameters,
    layers: Mutex<LayersParameters>,
    context: Arc<Context>,
    transfer: StreamSpore,
    compute: StreamSpore,
    kernels: NvidiaKernels,
}

impl Model for Transformer {
    type Meta = Device;
    type Error = SafeTensorsError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, meta: Self::Meta) -> Result<Self, Self::Error> {
        let context = Arc::new(meta.retain_primary());
        let time = Instant::now();
        let host = Memory::load_safetensors_realloc(
            model_dir,
            Some(|l| context.apply(|ctx| ctx.malloc_host::<u8>(l).sporulate())),
        )?;
        info!("load host: {:?}", time.elapsed());
        let load_layers = host.num_hidden_layers();

        let (model, layers, kernels, transfer, compute) = context.apply(|ctx| {
            let stream = ctx.stream();
            let block_size = ctx.dev().max_block_dims().0;
            (
                ModelParameters::new(&host, &stream),
                Mutex::new(LayersParameters::new(load_layers, &host, &stream)),
                NvidiaKernelsPtx::new(&host, block_size).load(ctx),
                stream.sporulate(),
                ctx.stream().sporulate(),
            )
        });

        Ok(Self {
            host,
            model,
            layers,
            context,
            transfer,
            compute,
            kernels,
        })
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn eos_token(&self) -> utok {
        self.host.eos_token_id()
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        let dt = self.host.data_type();
        let nlayers = self.host.num_hidden_layers() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let max_seq_len = self.host.max_position_embeddings() as udim;
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;

        Tensor::alloc(dt, &[nlayers, 2, nkvh, max_seq_len, d / nh], |len| Cache {
            context: self.context.clone(),
            mem: self.context.apply(|ctx| ctx.malloc::<u8>(len).sporulate()),
        })
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

        self.context.apply(|ctx| {
            let stream = ctx.stream();
            let mut ans = Tensor::alloc(cache.data_type(), cache.shape(), |len| {
                stream.malloc::<u8>(len)
            });
            let kernels = self.kernels.on(&stream);
            kernels.reform(
                &mut ans.as_mut().slice(&slice).map_physical(|u| &mut **u),
                &cache
                    .as_ref()
                    .slice(&slice)
                    .map_physical(|u| unsafe { u.mem.sprout(ctx) }),
            );
            ans.map_physical(|u| Cache {
                context: self.context.clone(),
                mem: u.sporulate(),
            })
        })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let dt = self.host.data_type();
        let d = self.host.hidden_size() as udim;
        self.context.apply(|ctx| {
            let compute = unsafe { self.compute.sprout(ctx) };
            let kernels = self.kernels.on(&compute);

            let tokens = queries.into_iter().collect::<Vec<_>>();
            let nt = tokens.len() as udim;

            let mut x = Tensor::alloc(dt, &[nt, d], |len| compute.malloc::<u8>(len));
            kernels.gather(&mut x, &self.host.embed_tokens(), tokens);
            x.map_physical(|u| Cache {
                context: self.context.clone(),
                mem: u.sporulate(),
            })
        })
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

        let dt = self.host.data_type();
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.host.intermediate_size() as udim;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();

        let mut x_ = token_embedded;
        self.context.apply(|ctx| {
            let compute = unsafe { self.compute.sprout(ctx) };
            let kernels = self.kernels.on(&compute);

            let reusing = (d + dkv + dkv).max(di + di);
            let mut state_buf = Tensor::alloc(dt, &[nt, d + reusing],|len| compute.malloc::<u8>(len));
            macro_rules! state {
                () => {
                    split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing)
                };
            }

            let mut q_buf = compute.malloc::<u8>((nh * max_seq_len * dh) as usize * dt.size());
            let mut att_buf = compute.malloc::<u8>((nh * max_seq_len * max_att_len) as usize * dt.size());
            let pos = causal_lm::pos(&queries, nt);
            let pos = pos.as_ref().map_physical(|u| compute.from_host(u));

            let mut x = x_.as_mut().map_physical(|u| unsafe { u.mem.sprout(ctx) });
            let transfer = unsafe { self.transfer.sprout(ctx) };
            // 层参数滚动加载是有状态的，必须由一个控制流独占。其他逻辑无状态，可以多流并发
            let mut layers = self.layers.lock().unwrap();
            for layer in 0..self.host.num_hidden_layers() {
                let params = {
                    layers.load(layer, &self.host, &transfer);
                    layers.sync(layer, &compute)
                };

                let (mut x1, qkv) = state!();
                let mut qkv = qkv.slice(&[slice![=>], slice![=> d + dkv + dkv]]);

                kernels.rms_norm(&mut x1, &x, &params.input_layernorm(ctx));
                kernels.mat_mul(&mut qkv, 0., &x1, &params.w_qkv(ctx), 1.);

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
                    let mut cache = query.cache.as_mut().map(|t|t.as_mut().map_physical(|u| unsafe { u.mem.sprout(ctx) }));
                    let mut  query = QueryContext{ cache:cache.as_mut(), range: query.range.clone() };
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

                kernels.mat_mul(&mut x, 1., &x1, &params.w_o(ctx), 1.);
                kernels.rms_norm(&mut x1, &x, &params.post_attention_layernorm(ctx));
                kernels.mat_mul(&mut gate_up, 0., &x1, &params.mlp_gate_up(ctx), 1.);
                let (mut gate, up) = split!(gate_up; [1]: di, di);
                kernels.swiglu(&mut gate, &up);
                kernels.mat_mul(&mut x, 1., &gate, &params.mlp_down(ctx), 1.);
            }
        });
        x_
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.host.data_type();
        let d = self.host.hidden_size();
        let voc = self.host.vocab_size() as udim;

        self.context.apply(|ctx| {
            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| unsafe { u.mem.sprout(ctx) });
            let compute = unsafe { self.compute.sprout(ctx) };
            let kernels = self.kernels.on(&compute);

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
            let dst_ = &mut **x.physical_mut();
            let src_ = unsafe { from_raw_parts(dst_.as_ptr(), dst_.len()) };
            for DecodingMeta {
                num_query,
                num_decode,
            } in iter
            {
                src += num_query - num_decode;
                if src > dst {
                    for _ in 0..num_decode {
                        compute
                            .memcpy_d2d(&mut dst_[dst * len..][..len], &src_[src * len..][..len]);
                        src += 1;
                        dst += 1;
                    }
                } else {
                    src += num_decode;
                    dst += num_decode;
                }
            }

            if dst == begin {
                return Tensor::alloc(dt, &[0, d as _], |_| Cache {
                    context: self.context.clone(),
                    mem: compute.malloc::<u8>(0).sporulate(),
                });
            }

            let mut x = x.slice(&[slice![begin => dst], slice![=>]]);
            let mut logits =
                Tensor::alloc(dt, &[x.shape()[0], voc], |len| compute.malloc::<u8>(len));

            let (model_norm, lm_head) = unsafe { self.model.release(&compute) };
            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            kernels.rms_norm(&mut x, &x_, &model_norm);
            kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);

            logits.map_physical(|u| Cache {
                context: self.context.clone(),
                mem: u.sporulate(),
            })
        })
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        mut logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        assert_eq!(logits.data_type(), DataType::F16);
        let &[_, voc] = logits.shape() else { panic!() };
        let voc = voc as usize;

        let mut host = vec![f16::ZERO; logits.size()];
        let Cache { context, mem } = logits.physical_mut();
        context.apply(|ctx| memcpy_d2h(&mut host, unsafe { &mem.sprout(ctx) }));

        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| args.random(&host[i * voc..][..voc]))
            .collect()
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        self.context.apply(|ctx| unsafe {
            self.model.kill(ctx);
            self.layers.lock().unwrap().kill(ctx);
            self.transfer.kill(ctx);
            self.compute.kill(ctx);
            self.kernels.kill(ctx);
        });
    }
}

pub struct Cache {
    pub context: Arc<Context>,
    pub mem: DevMemSpore,
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.context.apply(|ctx| unsafe { self.mem.kill(ctx) });
    }
}

#[test]
fn test_infer() {
    use std::time::Instant;

    let Some(model_dir) = common_nv::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    cuda::init();
    let Some(device) = cuda::Device::fetch() else {
        return;
    };

    let t0 = Instant::now();
    let model = <Transformer as Model>::load(model_dir, device).unwrap();
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
