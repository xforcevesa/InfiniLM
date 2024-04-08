use actix_web::{web, App, HttpServer};
use service::{Device, Service};
use std::sync::Arc;
use transformer::SampleArgs;

mod handlers;
mod manager;
mod response;
mod schemas;
use manager::ServiceManager;

/// All global variables and services shared among all endpoints in this App
pub struct AppState {
    /// Manager of this App, which provides all kinds of services such as infer, session management, etc
    pub service_manager: Arc<ServiceManager>,
}

pub fn create_app(model_path: &str, device: Device) -> web::Data<AppState> {
    let sample_args = SampleArgs {
        temperature: 1.0,
        top_k: 20,
        top_p: 1.0,
    };

    let infer_service = Arc::new(Service::load_model(model_path, sample_args, device));
    println!("Model loaded: {}", model_path);

    web::Data::new(AppState {
        service_manager: Arc::new(ServiceManager::from(infer_service)),
    })
}

pub async fn start_infer_service() -> std::io::Result<()> {
    let model_path = "/data1/shared/9G-Infer/models/11B-Chat-QY-epoch-8_F16";
    let ip = "127.0.0.1";
    let port = 5001;
    let device = Device::NvidiaGpu(7);
    
    let app_state = create_app(model_path, device);

    println!("Starting service at {}:{}", ip, port);
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(handlers::infer)
    })
    .bind((ip, port))?
    .run()
    .await
}
