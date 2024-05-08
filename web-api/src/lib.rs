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
use serde::Deserialize;
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

pub async fn start_infer_service<M>(
    service: service::Service<M>,
    port: u16,
    session_capacity: Option<usize>,
) -> std::io::Result<()>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let app = App(Arc::new(ServiceManager::new(service, session_capacity)));
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

struct App<M: CausalLM>(Arc<ServiceManager<M>>);

impl<M: CausalLM> Clone for App<M> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<M> hyper::service::Service<Request<body::Incoming>> for App<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<body::Incoming>) -> Self::Future {
        let manager = self.0.clone();

        fn parse_body<'a, S: Deserialize<'a>>(
            whole_body: &'a Bytes,
        ) -> Result<S, Response<BoxBody<Bytes, hyper::Error>>> {
            serde_json::from_slice(whole_body).map_err(|e| error(schemas::Error::WrongJson(e)))
        }

        match (req.method(), req.uri().path()) {
            (&Method::POST, "/infer") => Box::pin(async move {
                Ok(match parse_body(&req.collect().await?.to_bytes()) {
                    Ok(req) => match manager.infer(req) {
                        Ok(recv) => text_stream(UnboundedReceiverStream::new(recv)),
                        Err(e) => error(e),
                    },
                    Err(e) => e,
                })
            }),
            (&Method::POST, "/fork") => Box::pin(async move {
                Ok(match parse_body(&req.collect().await?.to_bytes()) {
                    Ok(req) => match manager.fork(req) {
                        Ok(s) => success(s),
                        Err(e) => error(e),
                    },
                    Err(e) => e,
                })
            }),
            (&Method::POST, "/drop") => Box::pin(async move {
                Ok(match parse_body(&req.collect().await?.to_bytes()) {
                    Ok(req) => match manager.drop_(req) {
                        Ok(s) => success(s),
                        Err(e) => error(e),
                    },
                    Err(e) => e,
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
