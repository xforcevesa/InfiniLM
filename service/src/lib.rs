mod session;
mod template;

use common::{upos, utok};
use half::f16;
use session::SessionComponent;
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        mpsc::{channel, Sender},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};
use template::Template;
use tensor::reslice;
use tokenizer::{Tokenizer, VocabTxt, BPE};
use transformer_cpu::{LayerCache, Llama2, Memory, Prompt, Request, Transformer};

pub use session::Session;

#[macro_use]
extern crate log;

pub struct Service {
    session_component: Arc<SessionComponent>,
    _manager: JoinHandle<()>,
}

impl Service {
    pub fn load_model(path: impl AsRef<Path>) -> Self {
        let model_dir = path.as_ref().to_owned();

        let template: Box<dyn Template> = {
            let path: String = model_dir.display().to_string();
            let path = path.to_ascii_lowercase();
            if path.contains("tinyllama") {
                Box::new(template::ChatTinyLlama)
            } else {
                Box::new(template::ChatCPM)
            }
        };

        let time = Instant::now();
        let tokenizer = tokenizer(&model_dir);
        info!("build tokenizer ... {:?}", time.elapsed());

        let (sender, receiver) = channel();
        Service {
            session_component: Arc::new(SessionComponent {
                template,
                tokenizer,
                sender,
            }),
            _manager: thread::spawn(move || {
                let time = Instant::now();
                let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
                info!("load model ... {:?}", time.elapsed());

                let eos = model.eos_token_id();
                let time = Instant::now();
                let mut transformer = Transformer::new(model);
                info!("build transformer ... {:?}", time.elapsed());

                let mut sessions = HashMap::new();

                while let Ok(cmd) = receiver.recv() {
                    match cmd {
                        Command::Chat {
                            id,
                            prompt,
                            responsing,
                        } => {
                            let ctx = sessions
                                .entry(id)
                                .or_insert_with(|| SessionContext::new(&transformer));

                            let time = Instant::now();
                            let (last, tokens) = prompt.split_last().expect("prompt is empty");
                            if !tokens.is_empty() {
                                transformer.decode(vec![Request {
                                    prompt: Prompt::Prefill(tokens),
                                    cache: &mut ctx.cache,
                                    pos: ctx.pos,
                                }]);
                            }
                            info!("prefill transformer ... {:?}", time.elapsed());

                            let mut token = *last;
                            let mut pos = tokens.len();
                            while pos < transformer.max_seq_len() {
                                let logits = transformer.decode(vec![Request {
                                    prompt: transformer_cpu::Prompt::Decode(token),
                                    cache: &mut ctx.cache,
                                    pos: pos as _,
                                }]);
                                token = argmax(reslice::<u8, f16>(logits.access().as_slice()));
                                responsing.send(token).unwrap();

                                if token == eos {
                                    break;
                                }

                                pos += 1;
                            }
                        }
                        Command::Drop { id } => {
                            sessions.remove(&id);
                        }
                    }
                }
            }),
        }
    }

    #[inline]
    pub fn launch(&self) -> Session {
        self.session_component.clone().into()
    }
}

enum Command {
    Chat {
        id: usize,
        prompt: Vec<utok>,
        responsing: Sender<utok>,
    },
    Drop {
        id: usize,
    },
}

struct SessionContext {
    pos: upos,
    cache: Vec<LayerCache>,
}

impl SessionContext {
    fn new(transformer: &Transformer) -> Self {
        Self {
            pos: 0,
            cache: transformer.new_cache(),
        }
    }
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(bpe) => return Box::new(bpe),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(voc) => return Box::new(voc),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}

fn argmax<T: PartialOrd>(logits: &[T]) -> utok {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as _
}
