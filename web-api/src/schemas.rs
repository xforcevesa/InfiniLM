#[derive(serde::Deserialize)]
pub struct InferRequest {
    pub session_id: String,
    pub inputs: String,
    pub first_request: bool,
}

#[derive(serde::Deserialize)]
pub struct CancelRequest {
    pub session_id: String,
}

pub trait Success {
    fn msg(&self) -> &str;
    fn extra(&self) -> Option<serde_json::Value> {
        None
    }
}

pub struct SessionCanceled;

impl Success for SessionCanceled {
    #[inline]
    fn msg(&self) -> &str {
        "Inferrence canceled"
    }
}

pub enum Error {
    SessionBusy,
    SessionNotFound,
}

impl Error {
    #[inline]
    pub fn msg(&self) -> &'static str {
        match self {
            Self::SessionBusy => "Session is busy",
            Self::SessionNotFound => "Session histroy is lost",
        }
    }
}
