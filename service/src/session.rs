use crate::{template::Template, Command};
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    mpsc::{channel, Sender},
    Arc,
};
use tokenizer::Tokenizer;

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
    pub fn chat(&self, prompt: &str, mut f: impl FnMut(&str)) {
        let prompt = self.component.template.encode(prompt);
        let prompt = self.component.tokenizer.encode(prompt.as_ref());

        let (responsing, receiver) = channel();
        let chat = Command::Chat {
            id: self.id,
            prompt,
            responsing,
        };

        self.component.sender.send(chat).unwrap();
        while let Ok(token) = receiver.recv() {
            let piece = self.component.tokenizer.decode(token);
            let piece = self.component.template.decode(piece);
            f(piece.as_ref());
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
    pub template: Box<dyn Template>,
    pub tokenizer: Box<dyn Tokenizer>,
    pub sender: Sender<Command>,
}
