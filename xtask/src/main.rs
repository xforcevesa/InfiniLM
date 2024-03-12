mod cast;
mod common;
mod generate;

use clap::Parser;

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Cast(cast) => cast.invode(),
        Generate(generate) => generate.invoke(),
        // Service(service) => service.launch(),
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
    // /// Start LLM inference service
    // Service(service::ServiceArgs),
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum Template {
    Chat9G,
    ChatTinyLlama,
}
