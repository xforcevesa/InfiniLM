use crate::InferenceArgs;
use service::Service;
use std::io::Write;

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    #[clap(flatten)]
    inference: InferenceArgs,
    /// Prompt.
    #[clap(short, long)]
    prompt: String,
}

impl GenerateArgs {
    pub fn invoke(self) {
        let service: Service = self.inference.into();

        print!("{}", self.prompt);
        service.launch().generate(&self.prompt, |piece| {
            print!("{piece}");
            std::io::stdout().flush().unwrap();
        });
    }
}
