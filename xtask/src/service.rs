#[derive(Args, Default)]
pub struct ServiceArgs {
    #[clap(flatten)]
    pub inference: crate::InferenceArgs,
    /// Port to bind the service to
    #[clap(short, long)]
    pub port: u16,
}

impl ServiceArgs {
    pub async fn serve(self) {
        use service::Service;
        use web_api::start_infer_service;

        macro_rules! serve {
            ($ty:ty; $meta:expr) => {
                let (mut service, _handle) = Service::<$ty>::load(&self.inference.model, $meta);
                service.default_sample = self.inference.sample_args();
                start_infer_service(service, self.port).await.unwrap();
            };
        }

        self.inference.init_log();
        match self.inference.nvidia().as_slice() {
            [] => {
                use transformer_cpu::Transformer as M;
                serve!(M; ());
            }
            #[cfg(feature = "nvidia")]
            &[n] => {
                use transformer_nv::{cuda, Transformer as M};
                serve!(M; cuda::Device::new(n));
            }
            #[cfg(feature = "nvidia")]
            _distribute => todo!(),
            #[cfg(not(feature = "nvidia"))]
            _ => panic!("Set \"nvidia\" feature to enablel nvidia support."),
        }
    }
}
