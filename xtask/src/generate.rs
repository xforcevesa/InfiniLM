use crate::{print_now, InferenceArgs, Task};
use causal_lm::CausalLM;
use service::Service;
use std::fmt::Debug;

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Prompt.
    #[clap(long, short)]
    pub prompt: String,
    /// Max number of steps to generate.
    #[clap(long)]
    pub max_steps: Option<usize>,
}

impl Task for GenerateArgs {
    fn inference(&self) -> &InferenceArgs {
        &self.inference
    }

    async fn typed<M>(self, meta: M::Meta)
    where
        M: CausalLM + Send + Sync + 'static,
        M::Storage: Send,
        M::Error: Debug,
    {
        let (service, _handle) = Service::<M>::load(&self.inference.model, meta);

        print_now!("{}", self.prompt);

        let mut steps = self.max_steps.unwrap_or(usize::MAX);
        let mut generator = service.generate(self.prompt, Some(self.inference.sample_args()));
        while let Some(s) = generator.decode().await {
            match &*s {
                "\\n" => println!(),
                _ => print_now!("{s}"),
            }
            steps -= 1;
            if steps == 0 {
                break;
            }
        }
        println!();
    }
}
