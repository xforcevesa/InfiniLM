use crate::schemas::{self, SessionCanceled};
use futures::{
    channel::mpsc::{self, Receiver},
    SinkExt,
};
use service::{Service, Session};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{atomic::AtomicBool, Arc, Mutex},
};

pub struct ServiceManager {
    /// Inference service provided by backend model
    infer_service: Service,

    /// All sessions, session id as key.
    /// New session will be created when a new id comes.
    /// The value will become empty when that session is being served,
    /// so that a new request with the same id will not be double-served.
    /// A session must be re-inserted after being served.
    sessions: Mutex<HashMap<String, Option<Session>>>,

    /// The abort flag for busy sessions, session id as key.
    cancel_flags: Mutex<HashMap<String, CancelFlag>>,
}

pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    #[inline]
    pub fn new() -> (Self, Self) {
        let flag = Arc::new(AtomicBool::new(false));
        (Self(flag.clone()), Self(flag))
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }

    #[inline]
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

impl From<Service> for ServiceManager {
    #[inline]
    fn from(infer_service: Service) -> Self {
        Self {
            infer_service,
            sessions: Default::default(),
            cancel_flags: Default::default(),
        }
    }
}

impl ServiceManager {
    /// Get existing or create new infer session for a infer request.
    /// Return session or error.
    pub fn register_inference(
        self: &Arc<Self>,
        request: schemas::InferRequest,
    ) -> Result<Receiver<String>, schemas::Error> {
        let mut session = match self
            .sessions
            .lock()
            .unwrap()
            .entry(request.session_id.clone())
        {
            // Case session id exists
            Entry::Occupied(mut e) => match e.get_mut().take() {
                // Session id exists but user thinks otherwise
                Some(_) if request.first_request => {
                    e.insert(Some(self.infer_service.launch())); // drop the old session
                    e.get_mut().take().unwrap()
                }
                // take the existing session
                Some(session) => session,
                // If session is being served
                None => return Err(schemas::Error::SessionBusy),
            },
            // First request, create new session
            Entry::Vacant(e) if request.first_request => {
                e.insert(Some(self.infer_service.launch())).take().unwrap()
            }
            // Session id does not exist but user thinks otherwise, histroy lost
            _ => return Err(schemas::Error::SessionNotFound),
        };

        let (flag_a, flag_b) = CancelFlag::new();
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(request.session_id.clone(), flag_a);

        let (mut sender, receiver) = mpsc::channel(4096);

        let self_ = self.clone();
        let session_id = request.session_id;
        tokio::spawn(async move {
            let mut busy = session.chat(&request.inputs);
            while let Some(s) = busy.decode().await {
                if flag_b.is_cancelled() {
                    break;
                }
                sender.send(s.into_owned()).await.unwrap();
            }
            self_.cancel_flags.lock().unwrap().remove(&session_id);
            self_
                .sessions
                .lock()
                .unwrap()
                .insert(session_id, Some(session));
        });

        Ok(receiver)
    }

    /// Signal the backend model to stop the current inferrence task given session id
    pub fn cancel_session(
        &self,
        request: schemas::CancelRequest,
    ) -> Result<SessionCanceled, schemas::Error> {
        self.cancel_flags
            .lock()
            .unwrap()
            .get(&request.session_id)
            .ok_or(schemas::Error::SessionNotFound)
            .map(|flag| {
                flag.cancel();
                SessionCanceled
            })
    }
}
