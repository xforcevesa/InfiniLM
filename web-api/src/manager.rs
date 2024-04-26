use crate::schemas::{Drop, DropSuccess, Error, Fork, ForkSuccess, Infer};
use causal_lm::CausalLM;
use futures::{
    channel::mpsc::{self, Receiver},
    SinkExt,
};
use service::{Service, Session};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Arc, Mutex},
};

pub struct ServiceManager<M: CausalLM> {
    /// Inference service provided by backend model
    infer_service: Service<M>,

    /// All sessions, session id as key.
    /// New session will be created when a new id comes.
    /// The value will become empty when that session is being served,
    /// so that a new request with the same id will not be double-served.
    /// A session must be re-inserted after being served.
    pending_sessions: Mutex<HashMap<String, Option<Session<M>>>>,
}

impl<M: CausalLM> From<Service<M>> for ServiceManager<M> {
    #[inline]
    fn from(infer_service: Service<M>) -> Self {
        Self {
            infer_service,
            pending_sessions: Default::default(),
        }
    }
}

impl<M> ServiceManager<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send + Sync + 'static,
{
    /// Get existing or create new infer session for a infer request.
    /// Return session or error.
    pub fn infer(
        self: &Arc<Self>,
        Infer {
            session_id,
            inputs,
            dialog_pos,
        }: Infer,
    ) -> Result<Receiver<String>, Error> {
        if inputs.is_empty() {
            return Err(Error::EmptyInput);
        }
        let mut session = match self
            .pending_sessions
            .lock()
            .unwrap()
            .entry(session_id.clone())
        {
            Entry::Occupied(mut e) => match e.get_mut() {
                Some(session) => {
                    if session.revert(dialog_pos).is_ok() {
                        Ok(e.get_mut().take().unwrap())
                    } else {
                        Err(Error::InvalidDialogPos(session.dialog_pos()))
                    }
                }
                None => Err(Error::SessionBusy),
            },
            Entry::Vacant(e) if dialog_pos == 0 => {
                Ok(e.insert(Some(self.infer_service.launch())).take().unwrap())
            }
            Entry::Vacant(_) => Err(Error::SessionNotFound),
        }?;
        let (mut sender, receiver) = mpsc::channel(4096);

        let self_ = self.clone();
        tokio::spawn(async move {
            {
                let mut busy = session.chat(inputs.iter().map(|s| s.content.as_str()));
                while let Some(s) = busy.decode().await {
                    if let Err(e) = sender.send(s.into_owned()).await {
                        warn!("Failed to send piece to {session_id} with error \"{e}\"");
                        break;
                    }
                }
            }
            if let Some(container) = self_.pending_sessions.lock().unwrap().get_mut(&session_id) {
                container.get_or_insert(session);
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
        let mut sessions = self.pending_sessions.lock().unwrap();
        if sessions.contains_key(&new_session_id) {
            warn!("Failed to fork because \"{new_session_id}\" already exists");
            return Err(Error::SessionDuplicate);
        }
        let new = sessions
            .get_mut(&session_id)
            .ok_or(Error::SessionNotFound)?
            .as_ref()
            .ok_or(Error::SessionBusy)?
            .fork();
        sessions.insert(new_session_id, Some(new));
        Ok(ForkSuccess)
    }

    pub fn drop_(&self, Drop { session_id }: Drop) -> Result<DropSuccess, Error> {
        self.pending_sessions
            .lock()
            .unwrap()
            .remove(&session_id)
            .map(|_| DropSuccess)
            .ok_or(Error::SessionNotFound)
    }
}
