mod cast;
mod chat;
mod service;

use causal_lm::SampleArgs;
use clap::Parser;
use service::ServiceArgs;
use std::{ffi::c_int, future::Future};

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Cast(cast) => cast.invode(),
        // Generate(args) => block_on(args.inference.generate(&args.prompt)),
        Chat(chat) => block_on(chat.chat()),
        Service(service) => block_on(service.serve()),
    }
}

#[inline]
fn block_on(f: impl Future) {
    #[cfg(feature = "nvidia")]
    {
        transformer_nv::cuda::init();
    }
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(f);
    runtime.shutdown_background();
    #[cfg(feature = "nvidia")]
    {
        transformer_nv::synchronize();
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
    /// Cast model
    Cast(cast::CastArgs),
    // /// Generate following text
    // Generate(generate::GenerateArgs),
    /// Chat locally
    Chat(InferenceArgs),
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
    /// Use Nvidia GPU.
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
