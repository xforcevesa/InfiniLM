mod cast;
mod chat;
mod deploy;
mod generate;
mod service;

use causal_lm::{CausalLM, SampleArgs};
use clap::Parser;
use deploy::DeployArgs;
use service::ServiceArgs;
use std::{ffi::c_int, fmt};

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Deploy(deploy) => deploy.deploy(),
        Cast(cast) => cast.invode(),
        Generate(args) => args.run(),
        Chat(chat) => chat.run(),
        Service(service) => service.run(),
    }
}

#[derive(Parser)]
#[clap(name = "transformer-utils")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Deploy binary
    Deploy(DeployArgs),
    /// Cast model
    Cast(cast::CastArgs),
    /// Generate following text
    Generate(generate::GenerateArgs),
    /// Chat locally
    Chat(chat::ChatArgs),
    /// Start the service
    Service(ServiceArgs),
}

#[derive(Args, Default)]
struct InferenceArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Random sample temperature.
    #[clap(long)]
    temperature: Option<f32>,
    /// Random sample top-k.
    #[clap(long)]
    top_k: Option<usize>,
    /// Random sample top-p.
    #[clap(long)]
    top_p: Option<f32>,

    #[cfg(feature = "nvidia")]
    /// Use Nvidia GPU, specify device IDs separated by comma, e.g. `0` or `0,1`.
    #[clap(long)]
    nvidia: Option<String>,
}

impl InferenceArgs {
    fn init_log(&self) {
        use log::LevelFilter;
        use simple_logger::SimpleLogger;

        let log = self
            .log
            .as_ref()
            .and_then(|log| match log.to_lowercase().as_str() {
                "off" | "none" => Some(LevelFilter::Off),
                "trace" => Some(LevelFilter::Trace),
                "debug" => Some(LevelFilter::Debug),
                "info" => Some(LevelFilter::Info),
                "error" => Some(LevelFilter::Error),
                _ => None,
            })
            .unwrap_or(LevelFilter::Warn);
        SimpleLogger::new().with_level(log).init().unwrap();
    }

    fn nvidia(&self) -> Vec<c_int> {
        self.nvidia
            .as_ref()
            .map_or("", String::as_str)
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<c_int>().unwrap())
            .collect()
    }

    #[inline]
    fn sample_args(&self) -> SampleArgs {
        SampleArgs {
            temperature: self.temperature.unwrap_or(0.),
            top_k: self.top_k.unwrap_or(usize::MAX),
            top_p: self.top_p.unwrap_or(1.),
        }
    }
}

trait Task: Sized {
    fn inference(&self) -> &InferenceArgs;

    async fn typed<M>(self, meta: M::Meta)
    where
        M: CausalLM + Send + Sync + 'static,
        M::Storage: Send,
        M::Error: fmt::Debug;

    fn run(self) {
        #[cfg(detected_cuda)]
        {
            transformer_nv::cuda::init();
        }
        let runtime = tokio::runtime::Runtime::new().unwrap();

        self.inference().init_log();
        match self.inference().nvidia().as_slice() {
            [] => {
                use transformer_cpu::Transformer as M;
                runtime.block_on(self.typed::<M>(()));
            }
            #[cfg(detected_cuda)]
            &[n] => {
                use transformer_nv::{cuda, Transformer as M};
                runtime.block_on(self.typed::<M>(cuda::Device::new(n)));
            }
            #[cfg(detected_nccl)]
            distribute => {
                use distributed::{cuda::Device, Transformer as M};
                let meta = distribute.iter().copied().map(Device::new).collect();
                runtime.block_on(self.typed::<M>(meta));
            }
            #[cfg(not(all(detected_cuda, detected_nccl)))]
            _ => panic!("Set \"nvidia\" feature to enablel nvidia support."),
        }

        runtime.shutdown_background();
        #[cfg(detected_cuda)]
        {
            transformer_nv::synchronize();
        }
    }
}

#[macro_export]
macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}
