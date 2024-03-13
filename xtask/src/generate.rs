use crate::{init_logger, service};
use std::io::Write;

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Prompt.
    #[clap(short, long)]
    prompt: String,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

impl GenerateArgs {
    pub fn invoke(self) {
        init_logger(self.log);
        let service = service(&self.model, self.nvidia);

        print!("{}", self.prompt);
        service.launch().generate(&self.prompt, |piece| {
            print!("{piece}");
            std::io::stdout().flush().unwrap();
        });
    }
}
