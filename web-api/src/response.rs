//! All HttpResponses in this App.

use std::convert::Infallible;

use crate::schemas;
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::{
    body::{Bytes, Frame},
    header::CONTENT_TYPE,
    Response, StatusCode,
};
use serde::Serialize;
use tokio_stream::{Stream, StreamExt};

pub fn text_stream(
    s: impl Stream<Item = String> + Send + Sync + 'static,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(stream(s))
        .unwrap()
}

pub fn success(success: impl schemas::Success) -> Response<BoxBody<Bytes, hyper::Error>> {
    #[derive(Serialize)]
    struct SuccessResponse<'a> {
        message: &'a str,
    }
    let body = SuccessResponse {
        message: success.msg(),
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .body(full(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

pub fn error(e: schemas::Error) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(e.status())
        .header(CONTENT_TYPE, "application/json")
        .body(full(serde_json::to_string(&e.body()).unwrap()))
        .unwrap()
}

#[inline]
fn stream<T>(chunk: T) -> BoxBody<Bytes, hyper::Error>
where
    T: Stream<Item = String> + Send + Sync + 'static,
{
    StreamBody::new(
        chunk.map(|s| Ok(Frame::data(s.into())).map_err(|_: Infallible| unreachable!())),
    )
    .boxed()
}

#[inline]
fn full<T: Into<Bytes>>(chunk: T) -> BoxBody<Bytes, hyper::Error> {
    Full::new(chunk.into())
        .map_err(|never| match never {})
        .boxed()
}
