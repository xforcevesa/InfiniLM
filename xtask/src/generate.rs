use crate::InferenceArgs;
use service::Service;
use std::io::Write;

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Prompt.
    #[clap(short, long)]
    pub prompt: String,
}

impl InferenceArgs {
    pub fn generate(self, prompt: &str) {
        let service: Service = self.into();

        print!("{prompt}");
        service.launch().generate(prompt, |piece| {
            print!("{piece}");
            std::io::stdout().flush().unwrap();
        });
    }
}
