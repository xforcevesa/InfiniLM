use clap::Parser;
use model_parameters::{save, DataType, Llama2, Memory};
use std::{fs, path::PathBuf, time::Instant};

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Cast(args) => args.cast(),
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
    Cast(CastArgs),
}

#[derive(Args, Default)]
struct CastArgs {
    /// Original model directory.
    #[clap(short, long)]
    model: String,
    /// Target model directory.
    #[clap(short, long)]
    target: Option<String>,
    /// Target model type.
    #[clap(short, long)]
    dt: Option<String>,
}

impl CastArgs {
    fn cast(self) {
        let ty = match self.dt.as_ref().map(String::as_str) {
            Some("f32") | Some("float") | Some("float32") | None => DataType::F32,
            Some("f16") | Some("half") | Some("float16") => DataType::F16,
            Some("bf16") | Some("bfloat16") => DataType::BF16,
            Some(ty) => panic!("Unknown data type: \"{ty}\""),
        };
        let model_dir = PathBuf::from(self.model);
        let time = Instant::now();
        let model = Memory::load_safetensors(&model_dir).unwrap();
        println!("load model ... {:?}", time.elapsed());
        if model.data_type() == ty {
            println!("Model already has target data type");
            return;
        }

        let target = self.target.map(PathBuf::from).unwrap_or_else(|| {
            model_dir.parent().unwrap().join(format!(
                "{}_{ty:?}",
                model_dir.file_name().unwrap().to_str().unwrap()
            ))
        });
        fs::create_dir_all(&target).unwrap();
        let t0 = Instant::now();
        let model = Memory::cast(&model, ty);
        let t1 = Instant::now();
        println!("cast data type ... {:?}", t1 - t0);
        save(&model, target).unwrap();
        let t2 = Instant::now();
        println!("save model ... {:?}", t2 - t1);
    }
}
