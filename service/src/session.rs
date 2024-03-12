use crate::{template::Template, Command};
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc,
};
use tokenizer::Tokenizer;
use tokio::{
    runtime::Builder,
    sync::mpsc::{channel, Sender},
};

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
    pub async fn chat(&self, prompt: &str, mut f: impl FnMut(&str)) {
        let prompt = self.component.template.encode(prompt);
        let prompt = self.component.tokenizer.encode(prompt.as_ref());

        let (responsing, mut receiver) = channel(64);
        let chat = Command::Chat {
            id: self.id,
            prompt,
            responsing,
        };

        self.component.sender.send(chat).await.unwrap();
        while let Some(token) = receiver.recv().await {
            let piece = self.component.tokenizer.decode(token);
            let piece = self.component.template.decode(piece);
            f(piece.as_ref());
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async move {
                self.component
                    .sender
                    .send(Command::Drop { id: self.id })
                    .await
                    .unwrap();
            });
    }
}

pub(crate) struct SessionComponent {
    pub template: Box<dyn Template>,
    pub tokenizer: Box<dyn Tokenizer>,
    pub sender: Sender<Command>,
}
