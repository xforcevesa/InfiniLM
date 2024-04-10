#![cfg(detected_nccl)]

mod gather;
mod parameters;

#[macro_use]
extern crate log;

pub use common_nv::cuda;

use common_nv::{
    cuda::{AsRaw, ContextResource, ContextSpore, DevMemSpore, Device, StreamSpore},
    udim, utok, Tensor,
};
use nccl::CommunicatorGroup;
use parameters::ParameterMatrix;
use std::{iter::zip, path::Path, time::Instant};
use transformer::{LayerCache, Llama2, Memory, Request};

pub struct Transformer {
    host: Memory,
    comms: CommunicatorGroup,
    matrix: ParameterMatrix,
}

impl transformer::Transformer for Transformer {
    type Cache = ();

    #[inline]
    fn model(&self) -> &dyn Llama2 {
        &self.host
    }

    fn new_cache(&self) -> Vec<LayerCache<Self::Cache>> {
        todo!()
    }

    fn decode<Id>(
        &self,
        mut requests: Vec<Request<Id, Self::Cache>>,
    ) -> (Vec<Id>, Tensor<Self::Cache>) {
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);

        let contexts = self.comms.contexts().collect::<Vec<_>>();
        let streams = contexts
            .iter()
            .map(|ctx| ctx.apply(|c| c.stream().sporulate()))
            .collect::<Vec<_>>();

        let x0 = self.token_embed(&requests, &streams);
        let x1 = zip(zip(&contexts, &streams), &x0)
            .map(|((context, stream), x)| {
                context.apply(|ctx| {
                    let stream = unsafe { stream.sprout(ctx) };
                    Tensor::alloc(x.data_type(), x.shape(), |len| {
                        stream.malloc::<u8>(len).sporulate()
                    })
                })
            })
            .collect::<Vec<_>>();

        for ((context, mut stream), (mut x0, mut x1)) in zip(zip(contexts, streams), zip(x0, x1)) {
            context.apply(|ctx| unsafe {
                stream.kill(ctx);
                x0.physical_mut().kill(ctx);
                x1.physical_mut().kill(ctx);
            });
        }
        todo!()
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

        Self {
            comms: CommunicatorGroup::new(
                &dev.iter()
                    .map(|dev| unsafe { dev.as_raw() })
                    .collect::<Vec<_>>(),
            ),
            matrix: ParameterMatrix::load(
                &host,
                &dev.iter().map(Device::retain_primary).collect::<Vec<_>>(),
            ),
            host,
        }
    }

    fn token_embed<Id>(
        &self,
        requests: &[Request<Id, ()>],
        streams: &[StreamSpore],
    ) -> Vec<Tensor<DevMemSpore>> {
        let dt = self.host.data_type();
        let nt = requests.iter().map(Request::seq_len).sum::<udim>();
        let d = self.host.hidden_size() as udim;

        let mut x0 = self
            .comms
            .contexts()
            .zip(streams)
            .map(|(context, stream)| {
                context.apply(|ctx| {
                    Tensor::alloc(dt, &[nt, d], |len| {
                        unsafe { stream.sprout(ctx) }.malloc::<u8>(len).sporulate()
                    })
                })
            })
            .collect::<Vec<_>>();

        let tokens = requests.iter().flat_map(Request::tokens).copied();
        gather::gather(
            &mut x0,
            &self.host.embed_tokens(),
            tokens,
            &self.comms,
            streams,
        );

        x0
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let contexts = self.comms.contexts().collect::<Vec<_>>();
        unsafe { self.matrix.kill(&contexts) }
    }
}

#[test]
fn test() {
    use common_nv::cuda::{self, Device};
    use log::LevelFilter::Trace;
    use simple_logger::SimpleLogger;
    use transformer::Transformer as _;

    const N: usize = 1;

    cuda::init();
    if Device::count() < N {
        return;
    }

    SimpleLogger::new().with_level(Trace).init().unwrap();

    let time = Instant::now();
    let transformer = Transformer::new(
        "../../../TinyLlama-1.1B-Chat-v1.0_F16",
        &[Device::fetch().unwrap()],
    );
    info!("load {:?}", time.elapsed());

    transformer.decode(vec![Request::new(0, &[1, 2, 3], &mut [], 0, true)]);
}
