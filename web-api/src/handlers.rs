use crate::response::Response;
use crate::schemas;
use crate::AppState;
use actix_web::{post, web, Error, HttpResponse};
use futures::channel::mpsc;
use futures::stream::{Stream, StreamExt};
use service::Session;

#[post("/infer")]
async fn infer(
    app_state: web::Data<AppState>,
    request: web::Json<schemas::InferRequest>,
) -> HttpResponse {
    println!("Request from {}: infer", request.session_id);

    match app_state.service_manager.get_session(&request) {
        Ok(session) => {
            let infer_stream = create_infer_stream(session, request, app_state);
            Response::text_stream(infer_stream)
        }
        Err(e) => Response::error(e),
    }
}

fn create_infer_stream(
    mut session: Session,
    request: web::Json<schemas::InferRequest>,
    app_state: web::Data<AppState>,
) -> impl Stream<Item = Result<web::Bytes, Error>> {
    let (sender, receiver) = mpsc::channel(4096);

    tokio::spawn(async move {
        let id = request.session_id.clone();
        session_async_infer(&mut session, request, sender).await;
        app_state.service_manager.reset_session(&id, session);
    });

    receiver.map(|word| Ok(web::Bytes::from(word)))
}

async fn session_async_infer(
    session: &mut Session,
    request: web::Json<schemas::InferRequest>,
    mut sender: mpsc::Sender<String>,
) {
    session
        .chat(&request.inputs, |s| {
            sender
                .try_send(s.to_string())
                .expect("Failed to write data into output channel.")
        })
        .await
}
