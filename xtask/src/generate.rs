use log::LevelFilter;
use service::{Device, Service};
use simple_logger::SimpleLogger;
use std::{io::Write, path::PathBuf};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Prompt.
    #[clap(short, long)]
    prompt: String,
    /// Tokenizer file.
    #[clap(short, long)]
    tokenizer: Option<String>,
    /// Max steps.
    #[clap(short, long)]
    step: Option<usize>,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

impl GenerateArgs {
    pub fn invoke(self) {
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

        let model_dir = PathBuf::from(self.model);
        print!("{}", self.prompt);
        let service = Service::load_model(
            model_dir,
            if self.nvidia {
                Device::NvidiaGpu(0)
            } else {
                Device::Cpu
            },
        );

        let session = service.launch();
        session.generate(&self.prompt, |piece| {
            print!("{piece}");
            std::io::stdout().flush().unwrap();
        });
    }
}
