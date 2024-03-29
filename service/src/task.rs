use crate::{session, Command};
use common::utok;
use std::{
    collections::HashMap,
    sync::{mpsc::Receiver, Arc, Mutex},
    thread::{self, JoinHandle},
    time::Instant,
};
use transformer::{LayerCache, Request, SampleArgs, Transformer};

pub(crate) struct Task<T>
where
    T: Transformer,
{
    transformer: T,
    sessions: HashMap<usize, SessionContext<T::Cache>>,
    sample: Arc<Mutex<SampleArgs>>,
}

impl<T: Transformer> Task<T> {
    #[inline]
    pub fn new(transformer: T, sample: Arc<Mutex<SampleArgs>>) -> Self {
        Self {
            transformer,
            sessions: HashMap::new(),
            sample,
        }
    }

    fn invoke(&mut self, cmd: Command) {
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

                let max_seq_len = self.transformer.model().max_position_embeddings();
                let eos = self.transformer.model().eos_token_id();

                let t0 = Instant::now();
                let mut token = self.transformer.decode(
                    vec![ctx.request(&prompt, max_seq_len)],
                    &self.sample.lock().unwrap(),
                )[0]
                .1;
                let t1 = Instant::now();
                let mut len = 0;
                while token != eos {
                    responsing.send(token).unwrap();
                    token = self.transformer.decode(
                        vec![ctx.request(&[token], max_seq_len)],
                        &self.sample.lock().unwrap(),
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

impl<T> Task<T>
where
    T: Transformer + Send + 'static,
    T::Cache: Send + 'static,
{
    pub fn run(mut self, receiver: Receiver<Command>) -> Vec<JoinHandle<()>> {
        vec![thread::spawn(move || {
            for cmd in receiver {
                self.invoke(cmd);
            }
        })]
    }
}

struct SessionContext<Cache>(session::SessionContext<LayerCache<Cache>>);

impl<Cache> SessionContext<Cache> {
    #[inline]
    fn new<T>(transformer: &T, id: usize) -> Self
    where
        T: Transformer<Cache = Cache>,
    {
        Self(session::SessionContext::new(transformer.new_cache(), id))
    }

    #[inline]
    fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> Request<usize, Cache> {
        let pos = self.0.request(tokens, max_seq_len);
        Request::new(
            self.0.id,
            &self.0.cache_map[pos..],
            &mut self.0.cache,
            pos as _,
            true,
        )
    }
}
