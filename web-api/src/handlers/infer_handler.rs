use crate::{response, schemas, AppState};
use actix_web::{post, web, Error, HttpResponse};
use futures::{
    channel::mpsc,
    stream::{Stream, StreamExt},
    SinkExt,
};

#[post("/infer")]
pub async fn infer(
    app_state: web::Data<AppState>,
    request: web::Json<schemas::InferRequest>,
) -> HttpResponse {
    info!("Request from {}: infer", request.session_id);
    match app_state.service_manager.get_session(&request) {
        Ok(session) => response::text_stream(create_infer_stream(session, request, app_state)),
        Err(e) => response::error(e),
    }
}

fn create_infer_stream(
    mut session: service::Session,
    request: web::Json<schemas::InferRequest>,
    app_state: web::Data<AppState>,
) -> impl Stream<Item = Result<web::Bytes, Error>> {
    let (mut sender, receiver) = mpsc::channel(4096);

    tokio::spawn(async move {
        let mut busy = session.chat(&request.inputs);
        while let Some(s) = busy.receive().await {
            sender.send(s.into_owned()).await.unwrap();
        }
        app_state
            .service_manager
            .reset_session(&request.session_id, session);
    });

    receiver.map(|word| Ok(word.into()))
}
