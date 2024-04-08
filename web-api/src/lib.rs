use actix_web::{web, App, HttpServer};
use service::Service;
use std::{fmt::Debug, net::ToSocketAddrs, sync::Arc};

mod handlers;
mod manager;
mod response;
mod schemas;
use manager::ServiceManager;

#[macro_use]
extern crate log;

/// All global variables and services shared among all endpoints in this App
pub struct AppState {
    /// Manager of this App, which provides all kinds of services such as infer, session management, etc
    pub service_manager: Arc<ServiceManager>,
}

pub async fn start_infer_service(
    service: Service,
    addrs: impl ToSocketAddrs + Debug,
) -> std::io::Result<()> {
    println!("Starting service at {addrs:?}");
    let app_state = web::Data::new(AppState {
        service_manager: Arc::new(ServiceManager::from(Arc::new(service))),
    });
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(handlers::infer)
            .service(handlers::cancel)
    })
    .bind(addrs)?
    .run()
    .await
}
