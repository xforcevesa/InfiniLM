mod cast;
mod generate;

use clap::Parser;

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;

fn main() {
    // set env for POWERSHELL: `$env:RUST_LOG="INFO";`
    env_logger::init();

    use Commands::*;
    match Cli::parse().command {
        Cast(cast) => cast.invode(),
        Generate(generate) => generate.invoke(),
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
}
