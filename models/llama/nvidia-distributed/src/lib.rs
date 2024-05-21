#![cfg(detected_nccl)]

mod distribute;
mod parameters;

#[macro_use]
extern crate log;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{upos, utok, FileLoadError};
use common_nv::{
    cast_dt,
    cuda::{
        AsRaw, Context, ContextResource, ContextSpore, DevMem, DevMemSpore, Device, HostMemSpore,
        StreamSpore,
    },
    sample_nv, slice, split, udim, DataType, KernelRuntime, Kernels, LocalSplitable, NvidiaKernels,
    NvidiaKernelsPtx, Tensor,
};
use itertools::izip;
use llama::InferenceConfig;
use nccl::CommunicatorGroup;
use parameters::{Layer, ParameterMatrix};
use std::{
    iter::{repeat, zip},
    mem::{take, ManuallyDrop},
    path::Path,
    slice::from_raw_parts,
    sync::Arc,
    time::Instant,
};

pub use common_nv::cuda;

pub struct Transformer {
    config: InferenceConfig,

    comms: CommunicatorGroup,
    streams: Vec<StreamSpore>,
    kernels: Vec<NvidiaKernels>,

    embed_tokens: Tensor<HostMemSpore>,
    matrix: ParameterMatrix,
    lm_layernorm: Tensor<DevMemSpore>,
    lm_head: Tensor<DevMemSpore>,
}

impl Model for Transformer {
    type Meta = Vec<Device>;
    type Error = FileLoadError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, meta: Self::Meta) -> Result<Self, Self::Error> {
        let time = Instant::now();
        let host = llama::Storage::load_safetensors(model_dir)?;
        info!("load host: {:?}", time.elapsed());

        let kernels =
            NvidiaKernelsPtx::new(&meta, host.config.d as _, host.config.max_seq_len as _);

        let contexts = meta
            .iter()
            .map(|dev| {
                dev.set_mempool_threshold(u64::MAX);
                dev.retain_primary()
            })
            .collect::<Vec<_>>();
        let comms = CommunicatorGroup::new(
            &meta
                .iter()
                .map(|dev| unsafe { dev.as_raw() })
                .collect::<Vec<_>>(),
        );
        let matrix = ParameterMatrix::load(&host, &contexts);
        let (embed_tokens, lm_layernorm, lm_head) = comms.contexts().next().unwrap().apply(|ctx| {
            (
                host.embed_tokens.map_physical(|u| {
                    let mut host = ctx.malloc_host::<u8>(u.len());
                    host.clone_from_slice(&u);
                    host.sporulate()
                }),
                host.lm_layernorm
                    .map_physical(|u| ctx.from_host(&u).sporulate()),
                host.lm_head.map_physical(|u| ctx.from_host(&u).sporulate()),
            )
        });
        let streams = contexts
            .iter()
            .map(|context| context.apply(|ctx| ctx.stream().sporulate()))
            .collect::<Vec<_>>();
        let kernels = zip(&contexts, &streams)
            .map(|(context, stream)| {
                context.apply(|ctx| kernels.load(&unsafe { stream.sprout(ctx) }))
            })
            .collect();
        Ok(Self {
            comms,
            streams,
            kernels,

            embed_tokens,
            matrix,
            lm_layernorm,
            lm_head,

            config: host.config,
        })
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.config.max_seq_len
    }
    #[inline]
    fn eos_token(&self) -> utok {
        self.config.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        let dt = self.config.dt;
        let nlayers = self.config.nlayers;
        let nkvh = self.config.nkvh;
        let max_seq_len = self.config.max_seq_len;
        let d = self.config.d;
        let nh = self.config.nh;

        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        let n = contexts.len() as udim;
        Tensor::alloc(dt, &[nlayers, 2, nkvh / n, max_seq_len, d / nh], |len| {
            Cache {
                mem: contexts
                    .iter()
                    .map(|context| context.apply(|ctx| ctx.malloc::<u8>(len).sporulate()))
                    .collect(),
                contexts: contexts.clone(),
            }
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

        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        let mem = contexts
            .iter()
            .enumerate()
            .map(|(i, context)| {
                context.apply(|ctx| {
                    let stream = ctx.stream();
                    let mut ans = Tensor::alloc(cache.data_type(), cache.shape(), |len| {
                        stream.malloc::<u8>(len)
                    });
                    let kernels = self.kernels[i].on(&stream);
                    kernels.reform(
                        &mut ans.as_mut().slice(&slice).map_physical(|u| &mut **u),
                        &cache
                            .as_ref()
                            .slice(&slice)
                            .map_physical(|u| unsafe { u.mem[i].sprout(ctx) }),
                    );
                    ans.take_physical().sporulate()
                })
            })
            .collect();

        Tensor::new(cache.data_type(), cache.shape(), Cache { contexts, mem })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        let dt = self.config.dt;
        let d = self.config.d;

        let mut x = Tensor::alloc(dt, &[nt, d], |len| malloc_all(&contexts, len));
        contexts[0].apply(|ctx| {
            let stream = unsafe { ctx.sprout(&self.streams[0]) };
            let kernels = self.kernels[0].on(&stream);
            let mut x = x.as_mut().map_physical(|u| unsafe { ctx.sprout(&u[0]) });
            kernels.gather(&mut x, &self.embed_tokens, tokens);
        });
        for (i, comm) in self.comms.call().iter().enumerate() {
            contexts[i].apply(|ctx| {
                let stream = unsafe { ctx.sprout(&self.streams[i]) };
                let mut dst = unsafe { ctx.sprout(&x.physical_mut()[i]) };
                comm.broadcast(&mut dst, None, 0, &stream);
            });
        }
        x.map_physical(|mem| Cache { contexts, mem })
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        let queries = queries.into_iter().collect::<Vec<_>>();
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
        let seq_len = &seq_len;

        let dt = self.config.dt;
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;

        let n = self.comms.len() as udim;
        let reusing = (d + dkv + dkv).max(di + di);
        let pos = causal_lm::pos(&queries, nt);
        let pos = &pos;

        std::thread::scope(|s| {
            let _ = self
                .comms
                .iter()
                .enumerate()
                .map(|(i, comm)| {
                    let queries = queries
                        .iter()
                        .map(|q| {
                            (
                                q.cache
                                    .as_ref()
                                    .map(|t| t.as_ref().map_physical(|u| &u.mem[i])),
                                q.range.clone(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let x = token_embedded.as_ref().map_physical(|u| &u.mem[i]);

                    s.spawn(move || {
                        comm.device().retain_primary().apply(|ctx| {
                            let stream = unsafe { ctx.sprout(&self.streams[i]) };
                            let kernels = self.kernels[i].on(&stream);
                            let mut queries = queries
                                .into_iter()
                                .map(|(cache, range)| {
                                    (
                                        cache.map(|t| t.map_physical(|u| unsafe { ctx.sprout(u) })),
                                        range,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let mut queries = queries
                                .iter_mut()
                                .map(|(cache, range)| QueryContext {
                                    cache: cache.as_mut(),
                                    range: range.clone(),
                                })
                                .collect::<Vec<_>>();
                            let mut x = x.map_physical(|u| unsafe { ctx.sprout(u) });

                            let mut state_buf = Tensor::alloc(dt, &[nt, d + reusing / n], |len| {
                                ctx.malloc::<u8>(len)
                            });

                            let mut q_buf =
                                ctx.malloc::<u8>((nh / n * max_seq_len * dh) as usize * dt.size());
                            let mut att_buf = ctx.malloc::<u8>(
                                (nh / n * max_seq_len * max_att_len) as usize * dt.size(),
                            );
                            let pos = pos.as_ref().map_physical(|u| stream.from_host(u));

                            for layer in 0..self.config.nlayers as usize {
                                let params = self.matrix.get(layer, i, ctx);

                                self.self_att(
                                    &kernels,
                                    &mut queries,
                                    seq_len,
                                    &params,
                                    &mut x,
                                    &mut state_buf,
                                    &pos,
                                    &mut q_buf,
                                    &mut att_buf,
                                    i,
                                    layer,
                                    nt,
                                );
                                comm.all_reduce(
                                    x.physical_mut(),
                                    None,
                                    cast_dt(self.config.dt),
                                    nccl::ReduceType::ncclSum,
                                    &stream,
                                );

                                self.mlp(&kernels, &params, &mut x, &mut state_buf, i);
                                comm.all_reduce(
                                    x.physical_mut(),
                                    None,
                                    cast_dt(self.config.dt),
                                    nccl::ReduceType::ncclSum,
                                    &stream,
                                );
                            }

                            state_buf.take_physical().drop_on(&stream);
                            pos.take_physical().drop_on(&stream);
                            q_buf.drop_on(&stream);
                            att_buf.drop_on(&stream);
                        });
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>();
        });
        token_embedded
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.config.dt;
        let d = self.config.d;

        let contexts = Arc::new(vec![self.comms.contexts().next().unwrap()]);
        let ans = contexts[0].apply(|ctx| {
            let stream = unsafe { self.streams[0].sprout(ctx) };

            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| unsafe { u.mem[0].sprout(ctx) });
            let range = DecodingMeta::select(&mut x, decoding, |dst, src| {
                stream.memcpy_d2d(dst, src);
            });

            if range.is_empty() {
                return Tensor::alloc(dt, &[0, d as _], |_| Cache {
                    contexts: contexts.clone(),
                    mem: vec![stream.malloc::<u8>(0).sporulate()],
                });
            }

            let model_norm = self
                .lm_layernorm
                .as_ref()
                .map_physical(|u| unsafe { u.sprout(ctx) });
            let lm_head = self
                .lm_head
                .as_ref()
                .map_physical(|u| unsafe { u.sprout(ctx) });

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], |len| {
                stream.malloc::<u8>(len)
            });

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            let kernels = self.kernels[0].on(&stream);
            kernels.rms_norm(&mut x, &x_, &model_norm, self.config.epsilon);
            kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);

            logits.map_physical(|u| Cache {
                contexts: contexts.clone(),
                mem: vec![u.sporulate()],
            })
        });

        take(&mut ManuallyDrop::new(hidden_state.take_physical()).mem)
            .into_iter()
            .zip(self.comms.contexts())
            .enumerate()
            .for_each(|(i, (mut mem, context))| {
                context.apply(|ctx| unsafe { mem.kill_on(&self.streams[i].sprout(ctx)) });
            });

        ans
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        assert_eq!(logits.data_type(), DataType::F16);
        let &[_nt, voc] = logits.shape() else {
            panic!()
        };
        let voc = voc as usize;
        let Cache { contexts, mem } = logits.physical();

        contexts[0].apply(|ctx| {
            sample_nv(
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
                    .enumerate(),
                &unsafe { mem[0].sprout(ctx) },
                voc,
                &unsafe { self.streams[0].sprout(ctx) },
            )
        })
    }
}

impl Transformer {
    fn self_att(
        &self,
        kernels: &KernelRuntime,
        queries: &mut [QueryContext<DevMem>],
        seq_len: &[udim],
        params: &Layer,
        x: &mut Tensor<DevMem>,
        state_buf: &mut Tensor<DevMem>,
        pos: &Tensor<DevMem>,
        q_buf: &mut DevMem,
        att_buf: &mut DevMem,
        i: usize,
        layer: usize,
        nt: udim,
    ) {
        let dt = self.config.dt;
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let theta = self.config.theta;
        let epsilon = self.config.epsilon;

        let n = self.comms.len() as udim;
        let reusing = (d + dkv + dkv).max(di + di);

        let (mut x1, qkv) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing / n);
        let mut qkv = qkv.slice(&[slice![=>], slice![=> (d + dkv + dkv) / n]]);

        kernels.rms_norm(&mut x1, x, &params.input_layernorm(), epsilon);
        kernels.mat_mul(&mut qkv, 0., &x1, &params.w_qkv(), 1.);

        let (q, k, v) = split!(qkv; [1]: d / n, dkv / n, dkv / n);
        let mut q = q.reshape(&[nt, nh / n, dh]);
        let mut k = k.reshape(&[nt, nkvh / n, dh]);
        let v = v.reshape(&[nt, nkvh / n, dh]);
        let o = x1.reshape(&[nt, nh, dh]);
        let o = o.slice(&[slice![=>], slice![=> nh / n], slice![=>]]);

        kernels.rotary_embedding(&mut q, pos, theta);
        kernels.rotary_embedding(&mut k, pos, theta);

        let q = q.transpose(&[1, 0, 2]).split(1, seq_len);
        let k = k.transpose(&[1, 0, 2]).split(1, seq_len);
        let v = v.transpose(&[1, 0, 2]).split(1, seq_len);
        let o = o.transpose(&[1, 0, 2]).split(1, seq_len);

        for (query, q, k, v, mut o) in izip!(queries, q, k, v, o) {
            let pos = query.pos();
            let seq_len = query.seq_len();
            let att_len = query.att_len();
            let Some((mut k_cache, mut v_cache)) = query.cache(layer as _) else {
                continue;
            };

            let slice_cat = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
            let slice_att = &[slice![=>], slice![      => att_len], slice![=>]];
            let shape_q0 = &[nkvh / n * head_group, seq_len, dh];
            let shape_q1 = &[nkvh / n, head_group * seq_len, dh];
            let shape_att0 = &[nkvh / n, head_group * seq_len, att_len];
            let shape_att1 = &[nkvh / n * head_group, seq_len, att_len];

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

        let (x1, _) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing / n);

        let o = x1.as_ref().slice(&[slice![=>], slice![=> d/n as udim]]);
        let o = o.map_physical(|u| &**u);
        kernels.mat_mul(x, if i == 0 { 1. } else { 0. }, &o, &params.w_o(), 1.);
    }

    fn mlp(
        &self,
        kernels: &KernelRuntime,
        params: &Layer,
        x: &mut Tensor<DevMem>,
        state_buf: &mut Tensor<DevMem>,
        i: usize,
    ) {
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;
        let epsilon = self.config.epsilon;

        let n = self.comms.len() as udim;
        let reusing = (d + dkv + dkv).max(di + di);

        let (mut x1, gate_up) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing / n);
        let mut gate_up = gate_up.slice(&[slice![=>], slice![=> (di + di) / n]]);

        kernels.rms_norm(&mut x1, x, &params.post_att_layernorm(), epsilon);
        kernels.mat_mul(&mut gate_up, 0., &x1, &params.mlp_gate_up(), 1.);

        let (mut gate, up) = split!(gate_up; [1]: di / n, di / n);

        kernels.swiglu(&mut gate, &up);
        kernels.mat_mul(
            x,
            if i == 0 { 1. } else { 0. },
            &gate,
            &params.mlp_down(),
            1.,
        );
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let contexts = self.comms.contexts().collect::<Vec<_>>();
        unsafe {
            contexts[0].apply(|ctx| {
                ctx.kill(self.embed_tokens.physical_mut());
                ctx.kill(self.lm_layernorm.physical_mut());
                ctx.kill(self.lm_head.physical_mut());
            });
            self.matrix.kill(&contexts);
            for (context, stream, kernels) in izip!(contexts, &mut self.streams, &mut self.kernels)
            {
                context.apply(|ctx| {
                    stream.kill(ctx);
                    kernels.kill(ctx);
                });
            }
        }
    }
}

pub struct Cache {
    pub contexts: Arc<Vec<Context>>,
    pub mem: Vec<DevMemSpore>,
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        for (context, mem) in zip(&*self.contexts, &mut self.mem) {
            context.apply(|ctx| unsafe { mem.kill(ctx) });
        }
    }
}

fn malloc_all(contexts: &[Context], len: usize) -> Vec<DevMemSpore> {
    contexts
        .iter()
        .map(|context| context.apply(|ctx| ctx.malloc::<u8>(len).sporulate()))
        .collect()
}

#[test]
fn test_infer() {
    cuda::init();
    if cuda::Device::count() >= 2 {
        causal_lm::test_impl::<Transformer>(
            [0, 1].map(cuda::Device::new).into_iter().collect(),
            &[
                29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592,
                21106, 29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
            ],
        );
    }
}
