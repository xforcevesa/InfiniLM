mod cast;
mod generate;
mod service;

use ::service::{Device, Service};
use clap::Parser;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Cast(cast) => cast.invode(),
        Generate(generate) => generate.invoke(),
        Service(service) => service.invoke(),
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
    Service(service::ServiceArgs),
}

fn init_logger(log: Option<String>) {
    use log::LevelFilter;
    use simple_logger::SimpleLogger;
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
}

fn service(model_dir: &str, nvidia: bool) -> Service {
    Service::load_model(
        model_dir,
        if nvidia {
            Device::NvidiaGpu(0)
        } else {
            Device::Cpu
        },
    )
}
