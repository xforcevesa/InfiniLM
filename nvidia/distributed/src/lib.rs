#![cfg(detected_nccl)]

mod parameters;

#[macro_use]
extern crate log;

pub use common_nv::cuda;

use common_nv::{
    cast_dt,
    cuda::{memcpy_d2h, AsRaw, Context, ContextResource, ContextSpore, DevMemSpore, Device},
    slice, split, udim, utok, DataType, NvidiaKernels, NvidiaKernelsPtx, Tensor,
};
use half::f16;
use nccl::CommunicatorGroup;
use parameters::ParameterMatrix;
use std::{iter::zip, path::Path, slice::from_raw_parts, sync::Arc, time::Instant};
use transformer::{pos, Kernels, LayerBuffer, LayerCache, Llama2, Memory, Request};

pub struct Transformer {
    host: Memory,
    comms: CommunicatorGroup,
    kernels: Vec<NvidiaKernels>,
    model_norm: Tensor<DevMemSpore>,
    lm_head: Tensor<DevMemSpore>,
    matrix: ParameterMatrix,
}

impl transformer::Transformer for Transformer {
    type Cache = Cache;

    #[inline]
    fn max_position_embeddings(&self) -> usize {
        self.host.max_position_embeddings()
    }

    #[inline]
    fn eos_token(&self) -> utok {
        self.host.eos_token_id()
    }

    #[inline]
    fn new_cache(&self) -> Vec<LayerCache<Self::Cache>> {
        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        LayerCache::new_layers(&self.host, |dt, shape| {
            let &[nkvh, max_seq_len, d] = shape else {
                panic!()
            };
            Tensor::alloc(
                dt,
                &[nkvh / self.comms.len() as udim, max_seq_len, d],
                |len| Cache {
                    mem: contexts
                        .iter()
                        .map(|context| context.apply(|ctx| ctx.malloc::<u8>(len).sporulate()))
                        .collect(),
                    contexts: contexts.clone(),
                },
            )
        })
    }

    fn decode<Id>(
        &self,
        mut requests: Vec<Request<Id, Self::Cache>>,
    ) -> (Vec<Id>, Tensor<Self::Cache>) {
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);

        let dt = self.host.data_type();
        let nt = requests.iter().map(Request::seq_len).sum::<udim>();
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let dh = d / nh;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dkv = nkvh * dh;
        let head_group = nh / nkvh;
        let di = self.host.intermediate_size() as udim;
        let head_div = (dh as f32).sqrt().recip();
        let contexts = self.comms.contexts().collect::<Vec<_>>();
        let mut streams = contexts
            .iter()
            .map(|ctx| ctx.apply(|c| c.stream().sporulate()))
            .collect::<Vec<_>>();
        let n = contexts.len();

        // token embedding
        let mut x0 = Tensor::alloc(dt, &[nt, d], |len| malloc_all(&contexts, len));
        contexts[0].apply(|ctx| {
            let stream = unsafe { ctx.sprout(&streams[0]) };
            let kernels = self.kernels[0].on(&stream);
            let mut x = unsafe { x0.as_mut().map_physical(|u| ctx.sprout(&u[0])) };
            kernels.gather(
                &mut x,
                &self.host.embed_tokens(),
                requests.iter().flat_map(Request::tokens).copied(),
            );
        });
        for (i, comm) in self.comms.call().iter().enumerate() {
            contexts[i].apply(|ctx| {
                let stream = unsafe { ctx.sprout(&streams[i]) };
                let mut dst = unsafe { ctx.sprout(&x0.physical_mut()[i]) };
                comm.broadcast(&mut dst, None, 0, &stream);
            });
        }
        let mut x1 = Tensor::alloc(dt, &[nt, d], |len| malloc_all(&contexts, len));
        let LayerBuffer {
            qkv,
            gate_up,
            q_buf,
            att_buf,
        } = LayerBuffer::alloc(&self.host, &requests, |len| malloc_all(&contexts, len / n));
        let mut buf = LayerBuffer {
            qkv: {
                let &[a, b] = qkv.shape() else { panic!() };
                Tensor::new(dt, &[a, b / n as udim], qkv.take_physical())
            },
            gate_up: {
                let &[a, b] = gate_up.shape() else { panic!() };
                Tensor::new(dt, &[a, b / n as udim], gate_up.take_physical())
            },
            q_buf,
            att_buf,
        };
        // 生成位置张量
        let nt = x0.shape()[0]; // `nt` for number of tokens
        let pos_ = pos(&requests, nt);
        let mut pos = Tensor::new(
            DataType::U32,
            &[nt],
            contexts
                .iter()
                .enumerate()
                .map(|(i, context)| {
                    context.apply(|ctx| {
                        unsafe { ctx.sprout(&streams[i]) }
                            .from_host(&pos_)
                            .sporulate()
                    })
                })
                .collect::<Vec<_>>(),
        );

        for layer in 0..self.host.num_hidden_layers() {
            // before attention
            for (i, context) in contexts.iter().enumerate() {
                context.apply(|ctx| {
                    let stream = unsafe { ctx.sprout(&streams[i]) };
                    let kernels = self.kernels[i].on(&stream);
                    let params = self.matrix.get(layer, i, ctx);

                    let x0 = unsafe { x0.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut x1 = unsafe { x1.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut qkv = unsafe { buf.qkv.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    kernels.rms_norm(&mut x1, &x0, &params.input_layernorm());
                    kernels.mat_mul(&mut qkv, 0., &x1, &params.w_qkv(), 1.);
                });
            }
            let (q, k, v) =
                split!(buf.qkv.as_ref(); [1]: d / n as udim, dkv / n as udim, dkv / n as udim);
            let mut q = q.reshape(&[nt, nh / n as udim, dh]);
            let mut k = k.reshape(&[nt, nkvh / n as udim, dh]);
            let v = v.reshape(&[nt, nkvh / n as udim, dh]);
            for (i, context) in contexts.iter().enumerate() {
                context.apply(|ctx| {
                    let stream = unsafe { ctx.sprout(&streams[i]) };
                    let kernels = self.kernels[i].on(&stream);

                    let pos = unsafe { pos.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut q = unsafe { q.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut k = unsafe { k.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    kernels.rotary_embedding(&mut q, &pos);
                    kernels.rotary_embedding(&mut k, &pos);
                });
            }
            let o = &mut x1;
            // attention
            let q = q.as_ref().transpose(&[1, 0, 2]);
            let k = k.as_ref().transpose(&[1, 0, 2]);
            let v = v.as_ref().transpose(&[1, 0, 2]);
            let mut o = o
                .as_mut()
                .reshape(&[nt, nh, dh])
                .transpose(&[1, 0, 2])
                .slice(&[slice![=> nh/n as udim], slice![=>], slice![=>]]);

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
                let mut o = o.as_mut().slice(req_slice);

                let shape_att0 = &[nkvh / n as udim, head_group * seq_len, att_len];
                let shape_att1 = &[nkvh / n as udim * head_group, seq_len, att_len];

                let mut q_att = Tensor::new(dt, &[nh / n as udim, seq_len, dh], &mut *buf.q_buf);
                let mut att = Tensor::new(dt, shape_att0, &mut *buf.att_buf);
                let (k_cache, v_cache) = r.cache(layer);

                for (i, context) in contexts.iter().enumerate() {
                    context.apply(|ctx| {
                        let stream = unsafe { ctx.sprout(&streams[i]) };
                        let kernels = self.kernels[i].on(&stream);

                        let q = unsafe { q.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                        let k = unsafe { k.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                        let v = unsafe { v.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                        let mut o = unsafe { o.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                        let mut q_att =
                            unsafe { q_att.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                        let mut k_cache =
                            unsafe { k_cache.as_mut().map_physical(|u| ctx.sprout(&u.mem[i])) };
                        let mut v_cache =
                            unsafe { v_cache.as_mut().map_physical(|u| ctx.sprout(&u.mem[i])) };
                        let mut att = unsafe { att.as_mut().map_physical(|u| ctx.sprout(&u[i])) };

                        let mut k_cat =
                            k_cache.as_mut().slice(cat_slice).map_physical(|u| &mut **u);
                        let mut v_cat =
                            v_cache.as_mut().slice(cat_slice).map_physical(|u| &mut **u);
                        kernels.reform(&mut q_att, &q);
                        kernels.reform(&mut k_cat, &k);
                        kernels.reform(&mut v_cat, &v);

                        let q_att = q_att.reshape(&[nkvh / n as udim, head_group * seq_len, dh]);
                        let k_att = k_cache.slice(att_slice).transpose(&[0, 2, 1]);
                        let v_att = v_cache.slice(att_slice);

                        kernels.mat_mul(&mut att, 0., &q_att, &k_att, head_div);
                        let mut att = att.reshape(shape_att1);
                        kernels.softmax(&mut att);
                        let mut x2 = q_att;
                        let att = att.reshape(shape_att0);
                        kernels.mat_mul(&mut x2, 0., &att, &v_att, 1.);

                        kernels.reform(&mut o, &x2.reshape(&[nh / n as udim, seq_len, dh]));
                    });
                }
            }
            // after attention
            for (i, comm) in self.comms.call().iter().enumerate() {
                contexts[i].apply(|ctx| {
                    let stream = unsafe { ctx.sprout(&streams[i]) };
                    let kernels = self.kernels[i].on(&stream);
                    let params = self.matrix.get(layer, i, ctx);

                    let mut x0 = unsafe { x0.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    let o = x1.as_ref().slice(&[slice![=>], slice![=> d/n as udim]]);
                    let o = unsafe { o.map_physical(|u| ctx.sprout(&u[i])) };
                    kernels.mat_mul(&mut x0, if i == 0 { 1. } else { 0. }, &o, &params.w_o(), 1.);
                    comm.all_reduce(
                        x0.physical_mut(),
                        None,
                        cast_dt(self.host.data_type()),
                        nccl::ReduceType::ncclSum,
                        &stream,
                    );
                });
            }
            // for (i, context) in contexts.iter().enumerate() {
            //     context.apply(|ctx| {
            //         let stream = unsafe { ctx.sprout(&streams[i]) };
            //         stream.synchronize();
            //     })
            // }
            // std::process::exit(0);
            for (i, context) in contexts.iter().enumerate() {
                context.apply(|ctx| {
                    let stream = unsafe { ctx.sprout(&streams[i]) };
                    let kernels = self.kernels[i].on(&stream);
                    let params = self.matrix.get(layer, i, ctx);

                    let x0 = unsafe { x0.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut x1 = unsafe { x1.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut gate_up =
                        unsafe { buf.gate_up.as_mut().map_physical(|u| ctx.sprout(&u[i])) };

                    kernels.rms_norm(&mut x1, &x0, &params.post_att_layernorm());
                    kernels.mat_mul(&mut gate_up, 0., &x1, &params.mlp_gate_up(), 1.);
                });
            }
            let (mut gate, up) = split!(buf.gate_up; [1]: di / n as udim, di / n as udim);
            for (i, comm) in self.comms.call().iter().enumerate() {
                contexts[i].apply(|ctx| {
                    let stream = unsafe { ctx.sprout(&streams[i]) };
                    let kernels = self.kernels[i].on(&stream);
                    let params = self.matrix.get(layer, i, ctx);

                    let mut gate = unsafe { gate.as_mut().map_physical(|u| ctx.sprout(&u[i])) };
                    let up = unsafe { up.as_ref().map_physical(|u| ctx.sprout(&u[i])) };
                    let mut x0 = unsafe { x0.as_mut().map_physical(|u| ctx.sprout(&u[i])) };

                    kernels.swiglu(&mut gate, &up);
                    kernels.mat_mul(
                        &mut x0,
                        if i == 0 { 1. } else { 0. },
                        &gate,
                        &params.mlp_down(),
                        1.,
                    );
                    comm.all_reduce(
                        x0.physical_mut(),
                        None,
                        cast_dt(self.host.data_type()),
                        nccl::ReduceType::ncclSum,
                        &stream,
                    );
                });
            }
        }
        // decode
        if requests[0].decode() {
            let context = self.comms.contexts().next().unwrap();
            let logits = context.apply(|ctx| {
                let stream = unsafe { ctx.sprout(&streams[0]) };
                let kernels = self.kernels[0].on(&stream);

                let slice = {
                    let mut dst = unsafe { ctx.sprout(&x0.physical_mut()[0]) };
                    let dst = &mut *dst;
                    let src = unsafe { from_raw_parts(dst.as_ptr(), dst.len()) };

                    let (head, others) = requests.split_first().unwrap();
                    let begin = head.seq_len() as usize - 1;

                    let mut i_src = begin;
                    let mut i_dst = begin;
                    for r in others {
                        i_src += r.seq_len() as usize;
                        if r.decode() {
                            i_dst += 1;
                            if i_dst < i_src {
                                stream.memcpy_d2d(dst, src);
                            }
                        }
                    }
                    slice![from begin, until i_dst + 1]
                };
                let x = x0.as_ref().slice(&[slice, slice![=>]]);
                let mut x = unsafe { x.map_physical(|u| ctx.sprout(&u[0])) };

                let dt = self.host.data_type();
                let voc = self.host.vocab_size() as udim;

                let mut logits =
                    Tensor::alloc(dt, &[x.shape()[0], voc], |len| stream.malloc::<u8>(len));
                // 复制一个 x 以实现原地归一化
                let x_ = unsafe {
                    x.as_ref()
                        .map_physical(|u| from_raw_parts(u.as_ptr(), u.len()))
                };

                let model_norm =
                    unsafe { self.model_norm.as_ref().map_physical(|u| ctx.sprout(u)) };
                let lm_head = unsafe { self.lm_head.as_ref().map_physical(|u| ctx.sprout(u)) };

                kernels.rms_norm(&mut x, &x_, &model_norm);
                kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);
                logits.map_physical(|mem| mem.sporulate())
            });
            let logits = logits.map_physical(|mem| Cache {
                contexts: Arc::new(vec![context]),
                mem: vec![mem],
            });
            // kill
            for (i, context) in contexts.iter().enumerate() {
                context.apply(|ctx| unsafe {
                    ctx.kill(&mut streams[i]);
                    ctx.kill(&mut x0.physical_mut()[i]);
                    ctx.kill(&mut x1.physical_mut()[i]);
                    ctx.kill(&mut buf.qkv.physical_mut()[i]);
                    ctx.kill(&mut buf.gate_up.physical_mut()[i]);
                    ctx.kill(&mut buf.q_buf[i]);
                    ctx.kill(&mut buf.att_buf[i]);
                    ctx.kill(&mut pos.physical_mut()[i]);
                });
            }
            (requests.into_iter().map(Request::id).collect(), logits)
        } else {
            todo!()
        }
    }

    fn sample<Id>(
        &self,
        args: &transformer::SampleArgs,
        requests: Vec<Id>,
        logits: Tensor<Self::Cache>,
    ) -> Vec<(Id, utok)> {
        assert_eq!(logits.data_type(), DataType::F16);
        let &[_, voc] = logits.shape() else { panic!() };

        let mut host = vec![f16::ZERO; logits.size()];
        let Cache { contexts, mem } = logits.physical();
        contexts[0].apply(|ctx| memcpy_d2h(&mut host, unsafe { &mem[0].sprout(ctx) }));

        requests
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, args.random(&host[i * voc as usize..][..voc as usize])))
            .collect()
    }
}

impl Transformer {
    pub fn new(model_dir: impl AsRef<Path>, dev: &[Device]) -> Self {
        let time = Instant::now();
        let host = Memory::load_safetensors(model_dir).unwrap();
        info!("load host: {:?}", time.elapsed());

        let block_size = dev.iter().map(|dev| dev.max_block_dims().0).min().unwrap();
        let contexts = dev.iter().map(Device::retain_primary).collect::<Vec<_>>();
        let kernels = NvidiaKernelsPtx::new(&host, block_size);

        let comms = CommunicatorGroup::new(
            &dev.iter()
                .map(|dev| unsafe { dev.as_raw() })
                .collect::<Vec<_>>(),
        );
        let (model_norm, lm_head) = comms.contexts().next().unwrap().apply(|ctx| {
            (
                ctx.from_host(host.model_norm().as_slice()).sporulate(),
                ctx.from_host(host.lm_head().as_slice()).sporulate(),
            )
        });
        Self {
            comms,
            kernels: contexts
                .iter()
                .map(|context| context.apply(|ctx| kernels.load(ctx)))
                .collect(),
            matrix: ParameterMatrix::load(&host, &contexts),
            model_norm: host.model_norm().map_physical(|_| model_norm),
            lm_head: host.lm_head().map_physical(|_| lm_head).transpose(&[1, 0]),
            host,
        }
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let contexts = self.comms.contexts().collect::<Vec<_>>();
        unsafe {
            contexts[0].apply(|ctx| {
                ctx.kill(self.model_norm.physical_mut());
                ctx.kill(self.lm_head.physical_mut());
            });
            self.matrix.kill(&contexts);
            for (context, kernels) in zip(contexts, &mut self.kernels) {
                context.apply(|ctx| kernels.kill(ctx));
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
fn test() {
    use common_nv::cuda::{self, Device};
    use log::LevelFilter::Trace;
    use simple_logger::SimpleLogger;
    use transformer::Transformer as _;

    let Some(model_dir) = common_nv::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    const N: usize = 4;

    cuda::init();
    if Device::count() < N {
        return;
    }

    SimpleLogger::new().with_level(Trace).init().unwrap();

    let time = Instant::now();
    let transformer = Transformer::new(model_dir, &[Device::fetch().unwrap()]);
    info!("load {:?}", time.elapsed());

    let time = Instant::now();
    let mut cache = transformer.new_cache();
    info!("new cache: {:?}", time.elapsed());

    let time = Instant::now();
    transformer.decode(vec![Request::new(0, &[1, 2, 3], &mut cache, 0, true)]);
    info!("decode: {:?}", time.elapsed());
}
