use crate::{print_now, InferenceArgs};
use causal_lm::{CausalLM, SampleArgs};
use service::Service;
use std::{fmt::Debug, path::Path};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Prompt.
    #[clap(long, short)]
    pub prompt: String,
    /// Max number of steps to generate.
    #[clap(long, short)]
    pub max_steps: Option<usize>,
}

impl GenerateArgs {
    pub async fn generate(self) {
        macro_rules! generate {
            ($ty:ty; $meta:expr) => {
                generate::<$ty>(
                    &self.inference.model,
                    $meta,
                    &self.prompt,
                    self.max_steps.unwrap_or(usize::MAX),
                    self.inference.sample_args(),
                )
                .await;
            };
        }

        self.inference.init_log();
        match self.inference.nvidia().as_slice() {
            [] => {
                use transformer_cpu::Transformer as M;
                generate!(M; ());
            }
            #[cfg(detected_cuda)]
            &[n] => {
                use transformer_nv::{cuda, Transformer as M};
                generate!(M; cuda::Device::new(n));
            }
            #[cfg(detected_nccl)]
            _distribute => todo!(),
            #[cfg(not(all(detected_cuda, detected_nccl)))]
            _ => panic!("Set \"nvidia\" feature to enablel nvidia support."),
        }
    }
}

async fn generate<M>(
    model_dir: impl AsRef<Path>,
    meta: M::Meta,
    prompt: impl AsRef<str>,
    max_steps: usize,
    sample: SampleArgs,
) where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
    M::Error: Debug,
{
    let (mut service, _handle) = Service::<M>::load(model_dir, meta);
    service.default_sample = sample;
    let mut generator = service.generate(prompt, None);
    let mut steps = 0;
    while let Some(s) = generator.decode().await {
        match &*s {
            "\\n" => println!(),
            _ => print_now!("{s}"),
        }
        steps += 1;
        if steps >= max_steps {
            break;
        }
    }
    println!();
}
