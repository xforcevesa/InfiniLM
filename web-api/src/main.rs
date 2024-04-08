use web_handler::start_infer_service;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    start_infer_service().await
}
