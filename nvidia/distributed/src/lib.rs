#![cfg(detected_nccl)]

mod parameters;

#[macro_use]
extern crate log;

pub use common_nv::cuda;

use common_nv::{
    cuda::{AsRaw, Context, ContextResource, ContextSpore, DevMemSpore, Device, StreamSpore},
    slice, udim, utok, NvidiaKernels, NvidiaKernelsPtx, Tensor,
};
use nccl::CommunicatorGroup;
use parameters::ParameterMatrix;
use std::{iter::zip, path::Path, sync::Arc, time::Instant};
use transformer::{Kernels, LayerBuffer, LayerCache, Llama2, Memory, Request};

pub struct Transformer {
    host: Memory,
    comms: CommunicatorGroup,
    kernels: Vec<NvidiaKernels>,
    matrix: ParameterMatrix,
}

impl transformer::Transformer for Transformer {
    type Cache = Cache;

    #[inline]
    fn model(&self) -> &dyn Llama2 {
        &self.host
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
        let contexts = self.comms.contexts().collect::<Vec<_>>();
        let mut streams = contexts
            .iter()
            .map(|ctx| ctx.apply(|c| c.stream().sporulate()))
            .collect::<Vec<_>>();
        let n = contexts.len();

        // token embedding
        let mut x0 = {
            // 填充 num tokens 到 n 的整数倍，以使用 AllGather 同步诸卡
            let distributed = (nt as usize + n - 1) / n;
            let nt_padding = (distributed / n * n) as udim;
            let mut x0 = Tensor::alloc(dt, &[nt_padding, d], |len| {
                malloc_all(&contexts, &streams, len)
            });

            let d = d as usize * dt.size();
            let table = self.host.embed_tokens();
            let table = table.as_slice();

            let mut iter = requests
                .iter()
                .flat_map(Request::tokens)
                .copied()
                .map(|t| t as usize)
                .enumerate();

            for (i, comm) in self.comms.call().iter().enumerate() {
                contexts[i].apply(|ctx| {
                    let stream = unsafe { streams[i].sprout(ctx) };
                    let mut dst = unsafe { x0.physical_mut()[i].sprout(ctx) };
                    // 为每个卡拷贝部分 token
                    for _ in 0..distributed {
                        let Some((i, t)) = iter.next() else { break };
                        stream.memcpy_h2d(&mut dst[d * i..][..d], &table[d * t..][..d]);
                    }
                    comm.all_gather(&mut dst, None, &stream);
                });
            }
            // 截取有效部分
            x0.slice(&[slice![take nt], slice![all]])
        };
        let mut x1 = Tensor::alloc(x0.data_type(), x0.shape(), |len| {
            malloc_all(&contexts, &streams, len)
        });
        let LayerBuffer {
            qkv,
            gate_up,
            q_buf,
            att_buf,
        } = LayerBuffer::alloc(&self.host, &requests, |len| {
            malloc_all(&contexts, &streams, len / n)
        });
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

        for layer in 0..self.host.num_hidden_layers() {
            for (i, context) in contexts.iter().enumerate() {
                context.apply(|ctx| {
                    let stream = unsafe { streams[i].sprout(ctx) };
                    let x0 = unsafe { x0.as_mut().map_physical(|u| u[i].sprout(ctx)) };
                    let mut x1 = unsafe { x1.as_mut().map_physical(|u| u[i].sprout(ctx)) };
                    let mut qkv = unsafe { buf.qkv.as_mut().map_physical(|u| u[i].sprout(ctx)) };

                    let layer = self.matrix.get(layer, i, ctx);
                    let kernels = self.kernels[i].on(&stream);
                    kernels.rms_norm(&mut x1, &x0, &layer.input_layernorm());
                    kernels.mat_mul(&mut qkv, 0., &x1, &layer.w_qkv(), 1.);
                });
            }
        }

        // kill
        for (i, context) in contexts.iter().enumerate() {
            context.apply(|ctx| unsafe {
                streams[i].kill(ctx);
                x0.physical_mut()[i].kill(ctx);
                x1.physical_mut()[i].kill(ctx);
                buf.qkv.physical_mut()[i].kill(ctx);
                buf.gate_up.physical_mut()[i].kill(ctx);
                buf.q_buf[i].kill(ctx);
                buf.att_buf[i].kill(ctx);
            });
        }
        (
            requests.into_iter().map(Request::id).collect(),
            Tensor::new(
                common_nv::DataType::U8,
                &[],
                Cache {
                    contexts: Arc::new(vec![]),
                    mem: vec![],
                },
            ),
        )
    }

    fn sample<Id>(
        &self,
        _args: &transformer::SampleArgs,
        _requests: Vec<Id>,
        _logits: Tensor<Self::Cache>,
    ) -> Vec<(Id, utok)> {
        todo!()
    }
}

impl Transformer {
    pub fn new(model_dir: impl AsRef<Path>, dev: &[Device]) -> Self {
        let time = Instant::now();
        let host = Memory::load_safetensors_from_dir(model_dir).unwrap();
        info!("load host: {:?}", time.elapsed());

        let block_size = dev.iter().map(|dev| dev.max_block_dims().0).min().unwrap();
        let contexts = dev.iter().map(Device::retain_primary).collect::<Vec<_>>();
        let kernels = NvidiaKernelsPtx::new(&host, block_size);

        Self {
            comms: CommunicatorGroup::new(
                &dev.iter()
                    .map(|dev| unsafe { dev.as_raw() })
                    .collect::<Vec<_>>(),
            ),
            kernels: contexts
                .iter()
                .map(|context| context.apply(|ctx| kernels.load(ctx)))
                .collect(),
            matrix: ParameterMatrix::load(&host, &contexts),
            host,
        }
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let contexts = self.comms.contexts().collect::<Vec<_>>();
        unsafe {
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

fn malloc_all(contexts: &[Context], streams: &[StreamSpore], len: usize) -> Vec<DevMemSpore> {
    contexts
        .iter()
        .zip(streams)
        .map(|(context, stream)| {
            context.apply(|ctx| unsafe { stream.sprout(ctx) }.malloc::<u8>(len).sporulate())
        })
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

    const N: usize = 1;

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
