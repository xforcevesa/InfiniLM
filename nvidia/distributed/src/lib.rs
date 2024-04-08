#![cfg(detected_nccl)]

mod parameters;

#[macro_use]
extern crate log;

pub use common_nv::cuda;

use common_nv::{
    cuda::{AsRaw, ContextResource, ContextSpore, CudaDataType, Device},
    utok, DataType, Tensor,
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

        let contexts = self.comms.context_iter().collect::<Vec<_>>();
        let streams = contexts
            .iter()
            .map(|ctx| ctx.apply(|c| c.stream().sporulate()))
            .collect::<Vec<_>>();

        for (context, mut stream) in zip(contexts, streams) {
            context.apply(|ctx| unsafe { stream.kill(ctx) });
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
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let contexts = self.comms.context_iter().collect::<Vec<_>>();
        unsafe { self.matrix.kill(&contexts) }
    }
}

fn convert(dt: DataType) -> CudaDataType {
    match dt {
        DataType::F16 => CudaDataType::half,
        DataType::BF16 => CudaDataType::nv_bfloat16,
        DataType::F32 => CudaDataType::float,
        DataType::F64 => CudaDataType::double,
        _ => unreachable!(),
    }
}
