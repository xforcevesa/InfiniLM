//! All HttpResponses in this App.

use crate::schemas;
use actix_web::{web, Error, HttpResponse, HttpResponseBuilder};
use futures::Stream;
use serde::Serialize;

#[inline]
pub fn text_stream(
    stream: impl Stream<Item = Result<web::Bytes, Error>> + 'static,
) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(stream)
}

#[inline]
pub fn success(s: impl schemas::Success) -> HttpResponse {
    #[derive(Serialize)]
    struct SuccessResponse<'a> {
        message: &'a str,
    }

    HttpResponse::Ok()
        .content_type("application/json")
        .json(SuccessResponse { message: s.msg() })
}

#[inline]
pub fn error(e: schemas::Error) -> HttpResponse {
    HttpResponseBuilder::new(e.status())
        .content_type("application/json")
        .json(e.body())
}
