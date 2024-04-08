use crate::InferenceArgs;
use web_api::start_infer_service;

#[derive(Args, Default)]
pub(crate) struct ServiceArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Address to bind the service to
    #[clap(short, long)]
    pub addr: String,
}

impl ServiceArgs {
    pub async fn serve(self) {
        start_infer_service(self.inference.into(), self.addr)
            .await
            .unwrap();
    }
}
