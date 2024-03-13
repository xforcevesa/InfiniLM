use crate::{argmax, Command};
use common::{upos, utok};
use half::f16;
use std::{collections::HashMap, path::Path, time::Instant};
use tensor::reslice;
use transformer_cpu::{LayerCache, Llama2, Memory, Request, Transformer};

pub struct CpuTask {
    eos: utok,
    transformer: Transformer,
    sessions: HashMap<usize, SessionContext>,
}

impl CpuTask {
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        let time = Instant::now();
        let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
        info!("load model ... {:?}", time.elapsed());

        let eos = model.eos_token_id();
        let time = Instant::now();
        let transformer = Transformer::new(model);
        info!("build transformer ... {:?}", time.elapsed());

        let sessions = HashMap::new();
        Self {
            eos,
            transformer,
            sessions,
        }
    }

    pub fn invoke(&mut self, cmd: Command) {
        match cmd {
            Command::Chat {
                id,
                prompt,
                responsing,
            } => {
                let ctx = self
                    .sessions
                    .entry(id)
                    .or_insert_with_key(|&id| SessionContext::new(&self.transformer, id));

                let time = Instant::now();
                let mut logits = self.transformer.decode(vec![ctx.request(&prompt)]);
                info!("prefill transformer ... {:?}", time.elapsed());

                let max_seq_len = self.transformer.max_seq_len() as upos;
                while ctx.pos < max_seq_len {
                    let token = argmax(reslice::<u8, f16>(logits.access().as_slice()));
                    if token == self.eos {
                        break;
                    }
                    responsing.send(token).unwrap();

                    logits = self.transformer.decode(vec![ctx.request(&[token])]);
                }
            }
            Command::Drop { id } => {
                self.sessions.remove(&id);
            }
        }
    }
}

struct SessionContext {
    id: usize,
    pos: upos,
    cache: Vec<LayerCache>,
}

impl SessionContext {
    fn new(transformer: &Transformer, id: usize) -> Self {
        Self {
            id,
            pos: 0,
            cache: transformer.new_cache(),
        }
    }

    fn request<'a>(&'a mut self, tokens: &'a [utok]) -> Request<usize> {
        let pos = self.pos;
        self.pos += tokens.len() as upos;
        Request {
            id: self.id,
            tokens,
            cache: &mut self.cache,
            pos,
        }
    }
}
