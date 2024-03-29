use crate::template::Template;
use common::utok;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc::{channel, Sender},
        Arc,
    },
    time::Instant,
};
use tokenizer::{Normalizer, Tokenizer};
use transformer::{LayerCache, Request};

pub struct Session {
    id: usize,
    component: Arc<SessionComponent>,
}

impl Session {
    #[inline]
    pub(crate) fn new(component: Arc<SessionComponent>) -> Self {
        static ID_ROOT: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: ID_ROOT.fetch_add(1, Relaxed),
            component,
        }
    }

    #[inline]
    pub const fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn chat(&mut self, prompt: &str, f: impl FnMut(&str)) {
        self.send(&self.component.template.apply_chat(prompt), f)
    }

    #[inline]
    pub fn generate(&mut self, prompt: &str, f: impl FnMut(&str)) {
        self.send(&self.component.template.normalize(prompt), f)
    }

    fn send(&self, prompt: &str, mut f: impl FnMut(&str)) {
        let _stamp = Instant::now();

        let prompt = self.component.normalizer.encode(prompt);
        let prompt = self.component.tokenizer.encode(&prompt);

        let (responsing, receiver) = channel();
        let chat = Message::Infer(
            self.id,
            Box::new(Infer {
                _stamp,
                prompt,
                responsing,
            }),
        );

        self.component.sender.send(chat).unwrap();
        while let Ok(token) = receiver.recv() {
            let piece = self.component.tokenizer.decode(token);
            let piece = self.component.normalizer.decode(piece);
            f(&piece);
        }
    }
}

impl Drop for Session {
    #[inline]
    fn drop(&mut self) {
        self.component.sender.send(Message::Drop(self.id)).unwrap();
    }
}

pub(crate) enum Message {
    Infer(usize, Box<Infer>),
    Drop(usize),
}

pub(crate) struct Infer {
    pub _stamp: Instant,
    pub prompt: Vec<utok>,
    pub responsing: Sender<utok>,
}

pub(crate) struct SessionComponent {
    pub template: Box<dyn Template + Send + Sync>,
    pub normalizer: Box<dyn Normalizer + Send + Sync>,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub sender: Sender<Message>,
}

pub(crate) struct SessionContext<Cache> {
    /// 会话标识符。
    pub id: usize,
    /// 上文缓存。
    pub cache: Vec<LayerCache<Cache>>,
    /// 上文缓存对应的上文 token。
    pub cache_map: Vec<utok>,
}

impl<Cache> SessionContext<Cache> {
    #[inline]
    pub fn new(cache: Vec<LayerCache<Cache>>, id: usize) -> Self {
        Self {
            id,
            cache,
            cache_map: Vec::new(),
        }
    }

    pub fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> Request<usize, Cache> {
        let pos: usize;
        if self.cache_map.len() + tokens.len() > max_seq_len {
            pos = self.cache_map.len().min(16);
            if tokens.len() > max_seq_len / 2 {
                let tokens = &tokens[tokens.len() - max_seq_len / 2..];
                self.cache_map.truncate(pos);
                self.cache_map.extend_from_slice(tokens);
            } else {
                let tail_len = (self.cache_map.len() - pos).min(64);
                let tail = self.cache_map.len() - tail_len;
                self.cache_map.copy_within(tail.., pos);
                self.cache_map.truncate(pos + tail_len);
                self.cache_map.extend_from_slice(tokens);
            }
        } else {
            pos = self.cache_map.len();
            self.cache_map.extend_from_slice(tokens);
        };
        Request::new(
            self.id,
            &self.cache_map[pos..],
            &mut self.cache,
            pos as _,
            true,
        )
    }
}
