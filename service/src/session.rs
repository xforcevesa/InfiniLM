use crate::template::Template;
use common::utok;
use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        Arc, Mutex,
    },
    time::Instant,
};
use tokenizer::{Normalizer, Tokenizer};
use tokio::sync::mpsc::{
    unbounded_channel, UnboundedReceiver, UnboundedSender, WeakUnboundedSender,
};
use transformer::{LayerCache, Request};

pub struct Session {
    id: usize,
    component: Arc<SessionComponent>,
    abort_handle: Arc<Mutex<Option<WeakUnboundedSender<Respond>>>>,
}

impl Session {
    #[inline]
    pub(crate) fn new(component: Arc<SessionComponent>) -> Self {
        static ID_ROOT: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: ID_ROOT.fetch_add(1, Relaxed),
            component,
            abort_handle: Default::default(),
        }
    }

    #[inline]
    pub const fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn handle(&self) -> SessionHandle {
        SessionHandle {
            id: self.id,
            abort_handle: self.abort_handle.clone(),
        }
    }

    #[inline]
    pub fn chat(&mut self, prompt: &str) -> BusySession {
        self.send(&self.component.template.apply_chat(prompt))
    }

    #[inline]
    pub fn generate(&mut self, prompt: &str) -> BusySession {
        self.send(&self.component.template.normalize(prompt))
    }

    fn send(&mut self, prompt: &str) -> BusySession {
        let _stamp = Instant::now();

        let prompt = self.component.normalizer.encode(prompt);
        let prompt = self.component.tokenizer.encode(&prompt);

        let (responsing, receiver) = unbounded_channel();
        self.abort_handle
            .lock()
            .unwrap()
            .replace(responsing.downgrade());

        let chat = Command::Infer(
            self.id,
            Box::new(Infer {
                _stamp,
                prompt,
                responsing,
            }),
        );

        self.component.sender.send(chat).unwrap();
        BusySession {
            session: self,
            receiver,
        }
    }
}

impl Drop for Session {
    #[inline]
    fn drop(&mut self) {
        self.component.sender.send(Command::Drop(self.id)).unwrap();
    }
}

pub struct BusySession<'a> {
    session: &'a mut Session,
    receiver: UnboundedReceiver<Respond>,
}

impl BusySession<'_> {
    pub async fn receive(&mut self) -> Option<Cow<str>> {
        if let Some(Respond::Token(token)) = self.receiver.recv().await {
            let SessionComponent {
                normalizer,
                tokenizer,
                ..
            } = &*self.session.component;
            Some(normalizer.decode(tokenizer.decode(token)))
        } else {
            None
        }
    }
}

pub struct SessionHandle {
    id: usize,
    abort_handle: Arc<Mutex<Option<WeakUnboundedSender<Respond>>>>,
}

impl SessionHandle {
    #[inline]
    pub const fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn abort(&self) {
        if let Some(sender) = self
            .abort_handle
            .lock()
            .unwrap()
            .as_ref()
            .and_then(WeakUnboundedSender::upgrade)
        {
            let _ = sender.send(Respond::Abort);
        }
    }
}

pub(crate) enum Command {
    Infer(usize, Box<Infer>),
    Drop(usize),
}

pub(crate) enum Respond {
    Token(utok),
    Abort,
}

pub(crate) struct Infer {
    pub _stamp: Instant,
    pub prompt: Vec<utok>,
    pub responsing: UnboundedSender<Respond>,
}

pub(crate) struct SessionComponent {
    pub template: Box<dyn Template + Send + Sync>,
    pub normalizer: Box<dyn Normalizer + Send + Sync>,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub sender: UnboundedSender<Command>,
}

pub(crate) struct SessionContext<Cache> {
    /// 会话标识符。
    pub id: usize,
    /// 上文缓存。
    pub cache: Vec<LayerCache<Cache>>,
    /// 上文缓存对应的上文 token。
    pub cache_map: Vec<utok>,
    /// 当前已经计算过上下文缓存的 token 数量。
    pub progress: usize,
}

impl<Cache> SessionContext<Cache> {
    #[inline]
    pub fn new(cache: Vec<LayerCache<Cache>>, id: usize) -> Self {
        Self {
            id,
            cache,
            cache_map: Vec::new(),
            progress: 0,
        }
    }

    pub fn push(&mut self, tokens: &[utok], max_seq_len: usize) {
        if self.cache_map.len() + tokens.len() > max_seq_len {
            self.progress = self.progress.min(16);
            if tokens.len() > max_seq_len / 2 {
                let tokens = &tokens[tokens.len() - max_seq_len / 2..];
                self.cache_map.truncate(self.progress);
                self.cache_map.extend_from_slice(tokens);
            } else {
                let tail_len = (self.cache_map.len() - self.progress).min(64);
                let tail = self.cache_map.len() - tail_len;
                self.cache_map.copy_within(tail.., self.progress);
                self.cache_map.truncate(self.progress + tail_len);
                self.cache_map.extend_from_slice(tokens);
            }
        } else {
            self.cache_map.extend_from_slice(tokens);
        }
    }

    #[inline]
    pub fn request(&mut self, max_tokens: usize) -> Request<usize, Cache> {
        let mut tokens = &self.cache_map[self.progress..];
        let decode = tokens.len() <= max_tokens;
        if !decode {
            tokens = &tokens[..max_tokens];
        }

        let pos = self.progress;
        self.progress += tokens.len();

        Request::new(self.id, tokens, &mut self.cache, pos as _, decode)
    }
}
