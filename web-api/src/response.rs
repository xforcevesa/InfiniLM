//! All HttpResponses in this App.

use crate::schemas;
use actix_web::{web, Error, HttpResponse};
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
        result: &'a str,
        extra: Option<serde_json::Value>,
    }

    HttpResponse::Ok()
        .content_type("application/json")
        .json(SuccessResponse {
            result: s.msg(),
            extra: s.extra(),
        })
}

#[inline]
pub fn error(e: schemas::SessionError) -> HttpResponse {
    #[derive(Serialize)]
    struct ErrorResponse {
        error: String,
    }

    HttpResponse::Ok()
        .content_type("application/json")
        .json(ErrorResponse {
            error: e.msg().into(),
        })
}
