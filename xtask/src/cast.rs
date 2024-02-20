use std::{fs, path::PathBuf, time::Instant};
use tensor::DataType;
use transformer_cpu::model_parameters::{save, Llama2, Memory};

#[derive(Args, Default)]
pub(crate) struct CastArgs {
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
    pub fn invode(self) {
        let ty = match self.dt.as_deref() {
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

        let time = Instant::now();
        let model = Memory::cast(&model, ty);
        println!("cast data type ... {:?}", time.elapsed());

        let time = Instant::now();
        save(&model, &target).unwrap();
        println!("save model ... {:?}", time.elapsed());

        let tokenizer = model_dir.join("tokenizer.model");
        if tokenizer.is_file() {
            let time = Instant::now();
            fs::copy(&tokenizer, target.join("tokenizer.model")).unwrap();
            println!("copy tokenizer ... {:?}", time.elapsed());
        }
    }
}
