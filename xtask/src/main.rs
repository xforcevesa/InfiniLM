mod cast;
mod chat;
mod generate;

use ::service::{Device, Service};
use clap::Parser;
use transformer::SampleArgs;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Cast(cast) => cast.invode(),
        Generate(args) => args.inference.generate(&args.prompt),
        Chat(chat) => chat.chat(),
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
    #[clap(long, default_value = "0.0")]
    temperature: f32,
    /// Top-k for random sampling.
    #[clap(long, default_value = "usize::MAX")]
    top_k: usize,
    /// Top-p for random sampling.
    #[clap(long, default_value = "1.0")]
    top_p: f32,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,
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
                temperature,
                top_k,
                top_p,
            },
            if nvidia {
                Device::NvidiaGpu(0)
            } else {
                Device::Cpu
            },
        )
    }
}
