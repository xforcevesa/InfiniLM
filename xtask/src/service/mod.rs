mod channel;
mod cpu;
#[cfg(detected_cuda)]
mod nvidia;

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
        crate::common::logger_init(&self.log);
        if self.nvidia {
            #[cfg(detected_cuda)]
            {
                nvidia::run(self);
            }
            #[cfg(not(detected_cuda))]
            {
                panic!("Nvidia GPU is not available");
            }
        } else {
            cpu::run(self);
        }
    }
}
