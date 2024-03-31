mod cast;
mod chat;
mod generate;

use std::future::Future;

use ::service::{Device, Service};
use clap::Parser;
use transformer::SampleArgs;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Cast(cast) => cast.invode(),
        Generate(args) => block_on(args.inference.generate(&args.prompt)),
        Chat(chat) => block_on(chat.chat()),
    }
}

#[inline]
fn block_on(f: impl Future) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(f);
    runtime.shutdown_background();
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
    /// Cast model
    Cast(cast::CastArgs),
    /// Generate following text
    Generate(generate::GenerateArgs),
    /// Start service
    Chat(InferenceArgs),
}

#[derive(Args, Default)]
struct InferenceArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Temperature for random sampling.
    #[clap(long)]
    temperature: Option<f32>,
    /// Top-k for random sampling.
    #[clap(long)]
    top_k: Option<usize>,
    /// Top-p for random sampling.
    #[clap(long)]
    top_p: Option<f32>,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,
    #[cfg(feature = "nvidia")]
    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

impl From<InferenceArgs> for Service {
    fn from(args: InferenceArgs) -> Self {
        use log::LevelFilter;
        use simple_logger::SimpleLogger;

        let InferenceArgs {
            model,
            temperature,
            top_k,
            top_p,
            #[cfg(feature = "nvidia")]
            nvidia,
            log,
        } = args;

        let log = log
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

        Service::load_model(
            model,
            SampleArgs {
                temperature: temperature.unwrap_or(0.),
                top_k: top_k.unwrap_or(usize::MAX),
                top_p: top_p.unwrap_or(1.),
            },
            #[cfg(feature = "nvidia")]
            {
                if nvidia {
                    Device::NvidiaGpu(0)
                } else {
                    Device::Cpu
                }
            },
            #[cfg(not(feature = "nvidia"))]
            {
                Device::Cpu
            },
        )
    }
}
