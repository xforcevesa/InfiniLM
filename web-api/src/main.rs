use service::{Device, Service};
use transformer::SampleArgs;
use web_api::start_infer_service;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    start_infer_service(
        Service::load_model(
            "/data1/shared/9G-Infer/models/11B-Chat-QY-epoch-8_F16",
            SampleArgs {
                temperature: 0.,
                top_k: usize::MAX,
                top_p: 1.,
            },
            Device::NvidiaGpu(7),
        ),
        "localhost:5001",
    )
    .await
}
