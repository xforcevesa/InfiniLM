use crate::{argmax, Command};
use common::{upos, utok};
use half::f16;
use std::{collections::HashMap, path::Path, time::Instant};
use tensor::reslice;
use transformer_cpu::{LayerCache, Llama2, Memory, Prompt, Request, Transformer};

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
                    .or_insert_with(|| SessionContext::new(&self.transformer));

                let time = Instant::now();
                let (last, tokens) = prompt.split_last().expect("prompt is empty");
                if !tokens.is_empty() {
                    self.transformer.decode(vec![Request {
                        prompt: Prompt::Prefill(tokens),
                        cache: &mut ctx.cache,
                        pos: ctx.pos,
                    }]);
                }
                info!("prefill transformer ... {:?}", time.elapsed());

                ctx.pos += tokens.len() as upos;
                let mut token = *last;
                let max_seq_len = self.transformer.max_seq_len() as upos;
                while ctx.pos < max_seq_len {
                    let logits = self.transformer.decode(vec![Request {
                        prompt: transformer_cpu::Prompt::Decode(token),
                        cache: &mut ctx.cache,
                        pos: ctx.pos,
                    }]);
                    token = argmax(reslice::<u8, f16>(logits.access().as_slice()));
                    ctx.pos += 1;
                    if token == self.eos {
                        break;
                    }
                    responsing.send(token).unwrap();
                }
            }
            Command::Drop { id } => {
                self.sessions.remove(&id);
            }
        }
    }
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
