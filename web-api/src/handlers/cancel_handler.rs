use crate::{response, schemas, AppState};
use actix_web::{post, web, HttpResponse};

#[post("/cancel")]
pub async fn cancel(
    app_state: web::Data<AppState>,
    request: web::Json<schemas::CancelRequest>,
) -> HttpResponse {
    info!("Request from {}: cancel infer", request.session_id);
    match app_state.service_manager.cancel_session(&request) {
        Ok(s) => response::success(s),
        Err(e) => response::error(e),
    }
}
