use crate::schemas::{Drop, DropSuccess, Error, Fork, ForkSuccess, Infer};
use causal_lm::CausalLM;
use lru::LruCache;
use service::{Service, Session};
use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc::{self, UnboundedReceiver};

pub(crate) struct ServiceManager<M: CausalLM> {
    service: Service<M>,
    pending: Mutex<LruCache<String, Option<Session<M>>>>,
}

impl<M: CausalLM> ServiceManager<M> {
    #[inline]
    pub fn new(service: Service<M>, capacity: Option<usize>) -> Self {
        let cap =
            capacity.map(|c| NonZeroUsize::new(c).expect("Session capacity must be non-zero"));
        Self {
            service,
            pending: Mutex::new(cap.map(LruCache::new).unwrap_or_else(LruCache::unbounded)),
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
            inputs: messages,
            session_id,
            dialog_pos,
            temperature,
            top_k,
            top_p,
        }: Infer,
    ) -> Result<UnboundedReceiver<String>, Error> {
        let dialog_pos = dialog_pos.unwrap_or(0);
        let (sender, receiver) = mpsc::unbounded_channel();

        macro_rules! set_sample {
            ($session:expr) => {
                if let Some(temperature) = temperature {
                    $session.sample.temperature = temperature;
                }
                if let Some(top_k) = top_k {
                    $session.sample.top_k = top_k;
                }
                if let Some(top_p) = top_p {
                    $session.sample.top_p = top_p;
                }
            };
        }

        if let Some(session_id) = session_id {
            let mut lru = self.pending.lock().unwrap();
            let mut session = match lru.get_mut(&session_id) {
                Some(option) => {
                    if let Some(session) = option.as_mut() {
                        if session.revert(dialog_pos).is_ok() {
                            info!("Session {session_id} reverted to {dialog_pos}, inference ready");
                            Ok(option.take().unwrap())
                        } else {
                            let current = session.dialog_pos();
                            warn!(
                                "Session {session_id} failed to revert from {current} to {dialog_pos}"
                            );
                            Err(Error::InvalidDialogPos(current))
                        }
                    } else {
                        warn!("Session {session_id} busy");
                        Err(Error::SessionBusy)
                    }
                }
                None if dialog_pos == 0 => {
                    info!("Session {session_id} created, inference ready");
                    if let Some((out, _)) = lru.push(session_id.clone(), None) {
                        warn!("Session {out} dropped because LRU cache is full");
                    }
                    Ok(self.service.launch())
                }
                None => {
                    warn!("Session {session_id} not found");
                    Err(Error::SessionNotFound)
                }
            }?;

            session.extend(messages.iter().map(|s| s.content.as_str()));
            set_sample!(session);

            if session.dialog_pos() % 2 == 1 {
                let self_ = self.clone();
                tokio::spawn(async move {
                    {
                        let mut busy = session.chat();
                        while let Some(s) = busy.decode().await {
                            if let Err(e) = sender.send(s.into_owned()) {
                                warn!("Failed to send piece to {session_id} with error \"{e}\"");
                                break;
                            }
                        }
                    }
                    if let Some(container) = self_.pending.lock().unwrap().get_mut(&session_id) {
                        container.get_or_insert(session);
                    }
                });
            }
        } else if dialog_pos != 0 {
            warn!("Temporary session must be created with zero dialog position");
            return Err(Error::InvalidDialogPos(0));
        } else if messages.len() % 2 == 1 {
            info!("Temporary session created, inference ready");
            let mut session = self.service.launch();
            session.extend(messages.iter().map(|s| s.content.as_str()));
            set_sample!(session);

            tokio::spawn(async move {
                let mut busy = session.chat();
                while let Some(s) = busy.decode().await {
                    if let Err(e) = sender.send(s.into_owned()) {
                        warn!("Failed to send piece to temporary session with error \"{e}\"");
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
        let mut sessions = self.pending.lock().unwrap();
        if !sessions.contains(&new_session_id) {
            let new = sessions
                .get_mut(&session_id)
                .ok_or(Error::SessionNotFound)?
                .as_ref()
                .ok_or(Error::SessionBusy)?
                .fork();

            info!("Session \"{new_session_id}\" is forked from \"{session_id}\"");
            if let Some((out, _)) = sessions.push(new_session_id, Some(new)) {
                warn!("Session {out} dropped because LRU cache is full");
            }
            Ok(ForkSuccess)
        } else {
            warn!("Session fork failed because \"{new_session_id}\" already exists");
            Err(Error::SessionDuplicate)
        }
    }

    pub fn drop_(&self, Drop { session_id }: Drop) -> Result<DropSuccess, Error> {
        if self.pending.lock().unwrap().pop(&session_id).is_some() {
            info!("Session \"{session_id}\" dropped");
            Ok(DropSuccess)
        } else {
            Err(Error::SessionNotFound)
        }
    }
}
