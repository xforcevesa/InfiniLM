use crate::InferenceArgs;
use service::Service;
use std::{
    borrow::Cow,
    fs::File,
    io::{Read, Write},
    path::Path,
};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Prompt.
    #[clap(short, long)]
    pub prompt: String,
}

impl InferenceArgs {
    pub async fn generate(self, prompt: &str) {
        let service: Service = self.into();

        let path = Path::new(prompt);
        let prompt = if path.is_file() {
            let mut buf = String::new();
            File::open(path).unwrap().read_to_string(&mut buf).unwrap();
            Cow::Owned(buf)
        } else {
            Cow::Borrowed(prompt)
        };

        print!("{prompt}");
        service
            .launch()
            .generate(&prompt, |piece| {
                print!("{piece}");
                std::io::stdout().flush().unwrap();
            })
            .await;
    }
}
