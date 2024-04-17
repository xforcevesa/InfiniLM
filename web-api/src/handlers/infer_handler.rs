use crate::{response, schemas, AppState};
use actix_web::{post, web, HttpResponse};
use futures::stream::StreamExt;

#[post("/infer")]
pub async fn infer(
    app_state: web::Data<AppState>,
    request: web::Json<schemas::InferRequest>,
) -> HttpResponse {
    info!("Request from {}: infer", request.session_id);
    match app_state
        .service_manager
        .register_inference(request.into_inner())
    {
        Ok(stream) => response::text_stream(stream.map(|word| Ok(word.into()))),
        Err(e) => response::error(e),
    }
}
