use crate::schemas::{Drop, DropSuccess, Error, Fork, ForkSuccess, Infer};
use causal_lm::CausalLM;
use service::{Service, Session};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc::{self, UnboundedReceiver};

pub struct ServiceManager<M: CausalLM> {
    infer_service: Service<M>,
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
    M::Storage: Send,
{
    pub fn infer(
        self: &Arc<Self>,
        Infer {
            inputs,
            session_id,
            dialog_pos,
            temperature,
            top_k,
            top_p,
        }: Infer,
    ) -> Result<UnboundedReceiver<String>, Error> {
        if inputs.is_empty() {
            return Err(Error::EmptyInput);
        }
        let dialog_pos = dialog_pos.unwrap_or(0);
        let (sender, receiver) = mpsc::unbounded_channel();
        if let Some(session_id) = session_id {
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
            if let Some(temperature) = temperature {
                session.sample.temperature = temperature;
            }
            if let Some(top_k) = top_k {
                session.sample.top_k = top_k;
            }
            if let Some(top_p) = top_p {
                session.sample.top_p = top_p;
            }

            let self_ = self.clone();
            tokio::spawn(async move {
                {
                    let mut busy = session.chat(inputs.iter().map(|s| s.content.as_str()));
                    while let Some(s) = busy.decode().await {
                        if let Err(e) = sender.send(s.into_owned()) {
                            warn!("Failed to send piece to {session_id} with error \"{e}\"");
                            break;
                        }
                    }
                }
                if let Some(container) = self_.pending_sessions.lock().unwrap().get_mut(&session_id)
                {
                    container.get_or_insert(session);
                }
            });
        } else {
            let mut session = self.infer_service.launch();
            if let Some(temperature) = temperature {
                session.sample.temperature = temperature;
            }
            if let Some(top_k) = top_k {
                session.sample.top_k = top_k;
            }
            if let Some(top_p) = top_p {
                session.sample.top_p = top_p;
            }

            tokio::spawn(async move {
                let mut busy = session.chat(inputs.iter().map(|s| s.content.as_str()));
                while let Some(s) = busy.decode().await {
                    if let Err(e) = sender.send(s.into_owned()) {
                        warn!("Failed to send piece to temp session with error \"{e}\"");
                        break;
                    }
                }
            });
        }
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
