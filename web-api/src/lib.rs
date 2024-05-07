#![doc = include_str!("../README.md")]

mod manager;
mod response;
mod schemas;

use causal_lm::CausalLM;
use http_body_util::{combinators::BoxBody, BodyExt, Empty};
use hyper::{
    body::{self, Bytes},
    server::conn::http1,
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::TokioIo;
use manager::ServiceManager;
use response::{error, success, text_stream};
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    pin::Pin,
    sync::Arc,
};
use tokio::net::TcpListener;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[macro_use]
extern crate log;

struct AppState<M: CausalLM> {
    service_manager: Arc<ServiceManager<M>>,
}

impl<M> Clone for AppState<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    fn clone(&self) -> Self {
        Self {
            service_manager: self.service_manager.clone(),
        }
    }
}

pub async fn start_infer_service<M>(service: service::Service<M>, port: u16) -> std::io::Result<()>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let app = AppState {
        service_manager: Arc::new(service.into()),
    };

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        let (stream, _) = listener.accept().await?;
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}");
            }
        });
    }
}

impl<M> hyper::service::Service<Request<body::Incoming>> for AppState<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<body::Incoming>) -> Self::Future {
        let manager = self.service_manager.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, "/infer") => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body).unwrap();
                let ans = manager.infer(req);
                Ok(match ans {
                    Ok(recv) => text_stream(UnboundedReceiverStream::new(recv)),
                    Err(e) => error(e),
                })
            }),
            (&Method::POST, "/fork") => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body).unwrap();
                let ans = manager.fork(req);
                Ok(match ans {
                    Ok(s) => success(s),
                    Err(e) => error(e),
                })
            }),
            (&Method::POST, "/drop") => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body).unwrap();
                let ans = manager.drop_(req);
                Ok(match ans {
                    Ok(s) => success(s),
                    Err(e) => error(e),
                })
            }),
            // Return 404 Not Found for other routes.
            _ => Box::pin(async {
                Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(
                        Empty::<Bytes>::new()
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .unwrap())
            }),
        }
    }
}
