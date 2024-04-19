#![doc = include_str!("../README.md")]

mod manager;
mod response;
mod schemas;

use actix_web::{post, web, App, HttpResponse, HttpServer};
use futures::StreamExt;
use manager::ServiceManager;
use schemas::{Drop, Fork, Infer};
use std::{fmt::Debug, net::ToSocketAddrs, sync::Arc};

#[macro_use]
extern crate log;

/// All global variables and services shared among all endpoints in this App
struct AppState {
    /// Manager of this App, which provides all kinds of services such as infer, session management, etc
    service_manager: Arc<ServiceManager>,
}

pub async fn start_infer_service(
    service: service::Service,
    addrs: impl ToSocketAddrs + Debug,
) -> std::io::Result<()> {
    info!("start service at {addrs:?}");
    let app_state = web::Data::new(AppState {
        service_manager: Arc::new(service.into()),
    });
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(infer)
            .service(fork)
            .service(drop)
    })
    .bind(addrs)?
    .run()
    .await
}

#[post("/infer")]
async fn infer(app_state: web::Data<AppState>, request: web::Json<Infer>) -> HttpResponse {
    info!("Request from {}: infer", request.session_id);
    match app_state.service_manager.infer(request.into_inner()) {
        Ok(stream) => response::text_stream(stream.map(|word| Ok(word.into()))),
        Err(e) => response::error(e),
    }
}

#[post("/fork")]
async fn fork(app_state: web::Data<AppState>, request: web::Json<Fork>) -> HttpResponse {
    info!("Request from {}: fork", request.session_id);
    match app_state.service_manager.fork(request.into_inner()) {
        Ok(s) => response::success(s),
        Err(e) => response::error(e),
    }
}

#[post("/drop")]
async fn drop(app_state: web::Data<AppState>, request: web::Json<Drop>) -> HttpResponse {
    info!("Request from {}: drop", request.session_id);
    match app_state.service_manager.drop_(request.into_inner()) {
        Ok(s) => response::success(s),
        Err(e) => response::error(e),
    }
}
