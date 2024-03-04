mod channel;
mod chat;
mod cpu;
#[cfg(detected_cuda)]
mod nvidia;

use crate::{common::tokenizer, Template};
use channel::channel;
use std::{path::PathBuf, time::Instant};
use tokenizer::Tokenizer;

#[derive(Args, Default)]
pub(crate) struct ServiceArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Tokenizer file.
    #[clap(short, long)]
    tokenizer: Option<String>,
    /// Channel type.
    #[clap(long)]
    channel: Option<String>,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

impl ServiceArgs {
    pub fn launch(self) {
        if self.nvidia {
            #[cfg(detected_cuda)]
            {
                nvidia::run(self.into());
            }
            #[cfg(not(detected_cuda))]
            {
                panic!("Nvidia GPU is not available");
            }
        } else {
            cpu::run(self.into());
        }
    }
}

struct ServiceParts {
    model_dir: PathBuf,
    template: Template,
    tokenizer: Box<dyn Tokenizer>,
    channel: Box<dyn channel::Channel>,
}

impl From<ServiceArgs> for ServiceParts {
    fn from(args: ServiceArgs) -> Self {
        crate::common::logger_init(&args.log);

        let template = if args.model.to_ascii_lowercase().contains("tinyllama") {
            Template::ChatTinyLlama
        } else {
            Template::Chat9G
        };
        let model_dir = PathBuf::from(&args.model);

        let time = Instant::now();
        let tokenizer = tokenizer(args.tokenizer, &model_dir);
        info!("build tokenizer ... {:?}", time.elapsed());

        let time = Instant::now();
        let channel = channel(args.channel);
        info!("build channel ... {:?}", time.elapsed());

        Self {
            model_dir,
            template,
            tokenizer,
            channel,
        }
    }
}
