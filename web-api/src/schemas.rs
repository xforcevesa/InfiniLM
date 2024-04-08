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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CancelRequest {
    pub session_id: String,
}

impl From<web::Json<CancelRequest>> for CancelRequest {
    fn from(request: web::Json<CancelRequest>) -> Self {
        CancelRequest {
            session_id: request.session_id.clone(),
        }
    }
}

pub enum Success {
    SessionCanceled,
}

impl Success {
    pub fn msg(&self) -> String {
        match self {
            Success::SessionCanceled => "Inferrence canceled".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SuccessResponse {
    pub result: String,
}

pub enum Error {
    SessionBusy,
    SessionNotFound,
    CancelFailed,
}

impl Error {
    pub fn msg(&self) -> String {
        match self {
            Error::SessionBusy => "Session is busy".to_string(),
            Error::SessionNotFound => "Session histroy is lost".to_string(),
            Error::CancelFailed => "Failed to cancel inferrence".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ErrorResponse {
    pub error: String,
}
