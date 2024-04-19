use crate::schemas::{Drop, DropSuccess, Error, Fork, ForkSuccess, Infer};
use futures::{
    channel::mpsc::{self, Receiver},
    SinkExt,
};
use service::{Service, Session};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Arc, Mutex},
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
}

impl From<Service> for ServiceManager {
    #[inline]
    fn from(infer_service: Service) -> Self {
        Self {
            infer_service,
            sessions: Default::default(),
        }
    }
}

impl ServiceManager {
    /// Get existing or create new infer session for a infer request.
    /// Return session or error.
    pub fn infer(
        self: &Arc<Self>,
        Infer {
            session_id,
            inputs,
            first_request,
        }: Infer,
    ) -> Result<Receiver<String>, Error> {
        let mut session = match self.sessions.lock().unwrap().entry(session_id.clone()) {
            // Case session id exists
            Entry::Occupied(mut e) => match e.get_mut().take() {
                // Session id exists but user thinks otherwise
                Some(_) if first_request => {
                    e.insert(Some(self.infer_service.launch())); // drop the old session
                    e.get_mut().take().unwrap()
                }
                // take the existing session
                Some(session) => session,
                // If session is being served
                None => return Err(Error::SessionBusy),
            },
            // First request, create new session
            Entry::Vacant(e) if first_request => {
                e.insert(Some(self.infer_service.launch())).take().unwrap()
            }
            // Session id does not exist but user thinks otherwise, histroy lost
            _ => return Err(Error::SessionNotFound),
        };

        let (mut sender, receiver) = mpsc::channel(4096);

        let self_ = self.clone();
        tokio::spawn(async move {
            let mut busy = session.chat(&inputs);
            while let Some(s) = busy.decode().await {
                if let Err(e) = sender.send(s.into_owned()).await {
                    warn!("Failed to send piece to {session_id} with error \"{e}\"");
                    break;
                }
            }
            if let Some(opt) = self_.sessions.lock().unwrap().get_mut(&session_id) {
                opt.get_or_insert(session);
            }
        });

        Ok(receiver)
    }

    pub fn fork(
        &self,
        Fork {
            session_id,
            new_session_id,
        }: Fork,
    ) -> Result<ForkSuccess, Error> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(&session_id)
            .ok_or(Error::SessionNotFound)?
            .take()
            .ok_or(Error::SessionBusy)?;
        let result = match sessions.entry(new_session_id) {
            Entry::Occupied(e) => {
                warn!("Failed to fork because \"{}\" already exists", e.key());
                Err(Error::SessionDuplicate)
            }
            Entry::Vacant(e) => {
                e.insert(Some(self.infer_service.fork(&session)));
                Ok(ForkSuccess)
            }
        };
        sessions.get_mut(&session_id).unwrap().replace(session);
        result
    }

    pub fn drop_(&self, Drop { session_id }: Drop) -> Result<DropSuccess, Error> {
        self.sessions
            .lock()
            .unwrap()
            .remove(&session_id)
            .map(|_| DropSuccess)
            .ok_or(Error::SessionNotFound)
    }
}
