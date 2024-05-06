use actix_web::http::StatusCode;

#[derive(serde::Deserialize)]
pub(crate) struct Infer {
    pub session_id: String,
    pub inputs: Vec<Sentence>,
    pub dialog_pos: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Sentence {
    #[allow(unused)]
    pub role: String,
    pub content: String,
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

#[derive(Debug)]
pub(crate) enum Error {
    SessionBusy,
    SessionDuplicate,
    SessionNotFound,
    EmptyInput,
    InvalidDialogPos(usize),
}

#[derive(serde::Serialize)]
struct ErrorBody {
    status: u16,
    code: u16,
    message: &'static str,
}

impl Error {
    #[inline]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::SessionNotFound => StatusCode::NOT_FOUND,
            Self::SessionBusy => StatusCode::NOT_ACCEPTABLE,
            Self::SessionDuplicate => StatusCode::CONFLICT,
            Self::EmptyInput => StatusCode::BAD_REQUEST,
            Self::InvalidDialogPos(_) => StatusCode::RANGE_NOT_SATISFIABLE,
        }
    }

    #[inline]
    pub fn body(&self) -> serde_json::Value {
        macro_rules! error {
            ($code:expr, $msg:expr) => {
                ErrorBody {
                    status: self.status().as_u16(),
                    code: $code,
                    message: $msg,
                }
            };
        }

        #[inline]
        fn json(v: impl serde::Serialize) -> serde_json::Value {
            serde_json::to_value(v).unwrap()
        }

        match self {
            Self::SessionNotFound => json(error!(0, "Session not found")),
            Self::SessionBusy => json(error!(0, "Session is busy")),
            Self::SessionDuplicate => json(error!(0, "Session ID already exists")),
            Self::EmptyInput => json(error!(0, "Input list is empty")),
            &Self::InvalidDialogPos(current_dialog_pos) => {
                #[derive(serde::Serialize)]
                struct ErrorBodyExtra {
                    #[serde(flatten)]
                    common: ErrorBody,
                    current_dialog_pos: usize,
                }
                json(ErrorBodyExtra {
                    common: error!(0, "Dialog position out of range"),
                    current_dialog_pos,
                })
            }
        }
    }
}
