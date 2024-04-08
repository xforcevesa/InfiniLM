use crate::schemas;
use actix_web::{web, Error, HttpResponse};
use futures::Stream;

/// All HttpResponses in this App
pub struct Response;

impl Response {
    pub fn text_stream(
        stream: impl Stream<Item = Result<web::Bytes, Error>> + 'static,
    ) -> HttpResponse {
        HttpResponse::Ok()
            .content_type("text/event-stream")
            .streaming(stream)
    }

    pub fn error(e: schemas::Error) -> HttpResponse {
        let err = schemas::ErrorResponse { error: e.msg() };
        HttpResponse::Ok()
            .content_type("application/json")
            .json(err)
    }

    pub fn success(s: schemas::Success) -> HttpResponse {
        let success = schemas::SuccessResponse { result: s.msg() };
        HttpResponse::Ok()
            .content_type("application/json")
            .json(success)
    }
}
