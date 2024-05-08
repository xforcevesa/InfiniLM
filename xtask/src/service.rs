use crate::{InferenceArgs, Task};
use causal_lm::CausalLM;
use service::Service;
use std::fmt::Debug;
use web_api::start_infer_service;

#[derive(Args, Default)]
pub struct ServiceArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Port to bind the service to
    #[clap(short, long)]
    pub port: u16,
    /// Maximum number of sessions to cache in memory.
    #[clap(long)]
    pub max_cache: Option<usize>,
}

impl Task for ServiceArgs {
    fn inference(&self) -> &InferenceArgs {
        &self.inference
    }

    async fn typed<M>(self, meta: M::Meta)
    where
        M: CausalLM + Send + Sync + 'static,
        M::Storage: Send,
        M::Error: Debug,
    {
        let (mut service, _handle) = Service::<M>::load(&self.inference.model, meta);
        service.default_sample = self.inference.sample_args();
        start_infer_service(service, self.port, self.max_cache.filter(|&c| c < 256))
            .await
            .unwrap();
    }
}
