use crate::{session, Command};
use common::utok;
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
use transformer_cpu::{LayerCache, Memory, Request, SampleArgs, Transformer};

pub struct CpuTask {
    transformer: Transformer,
    sessions: HashMap<usize, SessionContext>,
    sample: Arc<Mutex<SampleArgs>>,
}

impl CpuTask {
    pub fn new(model_dir: impl AsRef<Path>, sample: Arc<Mutex<SampleArgs>>) -> Self {
        let time = Instant::now();
        let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
        info!("load model ... {:?}", time.elapsed());

        let time = Instant::now();
        let transformer = Transformer::new(model);
        info!("build transformer ... {:?}", time.elapsed());

        let sessions = HashMap::new();
        Self {
            transformer,
            sessions,
            sample,
        }
    }

    pub fn invoke(&mut self, cmd: Command) {
        match cmd {
            Command::Infer {
                id,
                prompt,
                responsing,
            } => {
                let ctx = self
                    .sessions
                    .entry(id)
                    .or_insert_with_key(|&id| SessionContext::new(&self.transformer, id));

                let max_seq_len = self.transformer.max_seq_len();
                let eos = self.transformer.eos_token_id();

                let t0 = Instant::now();
                let mut token = self.transformer.decode(
                    vec![ctx.request(&prompt, max_seq_len)],
                    &*self.sample.lock().unwrap(),
                )[0]
                .1;
                let t1 = Instant::now();
                let mut len = 0;
                while token != eos {
                    responsing.send(token).unwrap();
                    token = self.transformer.decode(
                        vec![ctx.request(&[token], max_seq_len)],
                        &*self.sample.lock().unwrap(),
                    )[0]
                    .1;
                    len += 1;
                }
                let t2 = Instant::now();
                info!(
                    "First token delay: {:?}, average speed = {:?}/tok",
                    t1 - t0,
                    (t2 - t1).div_f32(len as _)
                );
            }
            Command::Drop { id } => {
                self.sessions.remove(&id);
            }
        }
    }
}

struct SessionContext(session::SessionContext<LayerCache>);

impl SessionContext {
    #[inline]
    fn new(transformer: &Transformer, id: usize) -> Self {
        Self(session::SessionContext::new(transformer.new_cache(), id))
    }

    #[inline]
    fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> Request<usize> {
        let pos = self.0.request(tokens, max_seq_len);
        Request {
            id: self.0.id,
            tokens: &self.0.cache_map[pos..],
            cache: &mut self.0.cache,
            pos: pos as _,
            decode: true,
        }
    }
}
