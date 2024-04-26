#![doc = include_str!("../README.md")]

mod manager;
mod response;
mod schemas;

use actix_web::{web, App, HttpResponse, HttpServer};
use causal_lm::CausalLM;
use futures::StreamExt;
use manager::ServiceManager;
use schemas::{Drop, Fork, Infer};
use std::{
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    sync::Arc,
};

#[macro_use]
extern crate log;

/// All global variables and services shared among all endpoints in this App
struct AppState<M: CausalLM> {
    /// Manager of this App, which provides all kinds of services such as infer, session management, etc
    service_manager: Arc<ServiceManager<M>>,
}

pub async fn start_infer_service<M>(service: service::Service<M>, port: u16) -> std::io::Result<()>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");
    let app_state = web::Data::new(AppState {
        service_manager: Arc::new(service.into()),
    });
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/infer", web::post().to(infer::<M>))
            .route("/fork", web::post().to(fork::<M>))
            .route("/drop", web::post().to(drop::<M>))
    })
    .bind(addr)?
    .run()
    .await
}

async fn infer<M>(app_state: web::Data<AppState<M>>, request: web::Json<Infer>) -> HttpResponse
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    info!("Request from {}: infer", request.session_id);
    match app_state.service_manager.infer(request.into_inner()) {
        Ok(stream) => response::text_stream(stream.map(|word| Ok(word.into()))),
        Err(e) => response::error(e),
    }
}

async fn fork<M>(app_state: web::Data<AppState<M>>, request: web::Json<Fork>) -> HttpResponse
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    info!("Request from {}: fork", request.session_id);
    match app_state.service_manager.fork(request.into_inner()) {
        Ok(s) => response::success(s),
        Err(e) => response::error(e),
    }
}

async fn drop<M>(app_state: web::Data<AppState<M>>, request: web::Json<Drop>) -> HttpResponse
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    info!("Request from {}: drop", request.session_id);
    match app_state.service_manager.drop_(request.into_inner()) {
        Ok(s) => response::success(s),
        Err(e) => response::error(e),
    }
}
