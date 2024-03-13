mod cast;
mod generate;
mod service;

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
