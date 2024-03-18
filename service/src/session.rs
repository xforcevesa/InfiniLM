use crate::{template::Template, Command};
use common::utok;
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    mpsc::{channel, Sender},
    Arc,
};
use tokenizer::{Normalizer, Tokenizer};

pub struct Session {
    id: usize,
    component: Arc<SessionComponent>,
}

impl From<Arc<SessionComponent>> for Session {
    fn from(component: Arc<SessionComponent>) -> Self {
        static ID_ROOT: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: ID_ROOT.fetch_add(1, Relaxed),
            component,
        }
    }
}

impl Session {
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
        let prompt = self.component.normalizer.encode(prompt);
        let prompt = self.component.tokenizer.encode(&prompt);

        let (responsing, receiver) = channel();
        let chat = Command::Infer {
            id: self.id,
            prompt,
            responsing,
        };

        self.component.sender.send(chat).unwrap();
        while let Ok(token) = receiver.recv() {
            let piece = self.component.tokenizer.decode(token);
            let piece = self.component.normalizer.decode(piece);
            f(&piece);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.component
            .sender
            .send(Command::Drop { id: self.id })
            .unwrap();
    }
}

pub(crate) struct SessionComponent {
    pub template: Box<dyn Template + Send + Sync>,
    pub normalizer: Box<dyn Normalizer + Send + Sync>,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub sender: Sender<Command>,
}

pub(crate) struct SessionContext<Cache> {
    pub id: usize,
    pub tokens: Vec<utok>,
    pub cache: Vec<Cache>,
}

impl<Cache> SessionContext<Cache> {
    #[inline]
    pub fn new(cache: Vec<Cache>, id: usize) -> Self {
        Self {
            id,
            tokens: Vec::new(),
            cache,
        }
    }

    #[inline]
    pub fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> usize {
        if self.tokens.len() + tokens.len() > max_seq_len {
            let pos = self.tokens.len().min(16);
            if tokens.len() > max_seq_len / 2 {
                let tokens = &tokens[tokens.len() - max_seq_len / 2..];
                self.tokens.truncate(pos);
                self.tokens.extend_from_slice(tokens);
            } else {
                let tail_len = (self.tokens.len() - pos).min(64);
                let tail = self.tokens.len() - tail_len;
                self.tokens.copy_within(tail.., pos);
                self.tokens.truncate(pos + tail_len);
                self.tokens.extend_from_slice(tokens);
            }
            pos
        } else {
            let pos = self.tokens.len();
            self.tokens.extend_from_slice(tokens);
            pos
        }
    }
}
