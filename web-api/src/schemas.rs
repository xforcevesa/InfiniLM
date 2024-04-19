#[derive(serde::Deserialize)]
pub(crate) struct Infer {
    pub session_id: String,
    pub inputs: String,
    pub first_request: bool,
}

#[derive(serde::Deserialize)]
pub(crate) struct Fork {
    pub session_id: String,
    pub new_session_id: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct Drop {
    pub session_id: String,
}

pub(crate) struct ForkSuccess;
pub(crate) struct DropSuccess;

pub(crate) trait Success {
    fn msg(&self) -> &str;
    fn extra(&self) -> Option<serde_json::Value> {
        None
    }
}

impl Success for ForkSuccess {
    fn msg(&self) -> &str {
        "fork success"
    }
}
impl Success for DropSuccess {
    fn msg(&self) -> &str {
        "drop success"
    }
}

pub(crate) enum Error {
    SessionBusy,
    SessionDuplicate,
    SessionNotFound,
}

impl Error {
    #[inline]
    pub fn msg(&self) -> &'static str {
        match self {
            Self::SessionBusy => "Session is busy",
            Self::SessionDuplicate => "Session already exists",
            Self::SessionNotFound => "Session histroy is lost",
        }
    }
}
