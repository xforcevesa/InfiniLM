use actix_web::web;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct InferRequest {
    pub session_id: String,
    pub inputs: String,
    pub first_request: bool,
}

impl From<web::Json<InferRequest>> for InferRequest {
    fn from(request: web::Json<InferRequest>) -> Self {
        InferRequest {
            session_id: request.session_id.clone(),
            inputs: request.inputs.clone(),
            first_request: request.first_request,
        }
    }
}

pub enum Error {
    SessionBusy,
    SessionNotFound,
}

impl Error {
    pub fn msg(&self) -> String {
        match self {
            Error::SessionBusy => "Session is busy".to_string(),
            Error::SessionNotFound => "Session histroy is lost".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ErrorResponse {
    pub error: String,
}
