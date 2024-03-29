use crate::{session, Command};
use common::utok;
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};
use transformer::{LayerCache, Request, SampleArgs, Transformer};

pub(crate) struct Task<T>
where
    T: Transformer,
{
    transformer: T,
    sample: Arc<Mutex<SampleArgs>>,
}

impl<T: Transformer> Task<T> {
    #[inline]
    pub fn new(transformer: T, sample: Arc<Mutex<SampleArgs>>) -> Self {
        Self {
            transformer,
            sample,
        }
    }
}

impl<T> Task<T>
where
    T: Transformer + Send + Sync + 'static,
    T::Cache: Send + 'static,
{
    pub fn run(self, receiver: Receiver<Command>) -> Vec<JoinHandle<()>> {
        struct Temp<T: Transformer> {
            ctx: SessionContext<T::Cache>,
            prompts: Vec<utok>,
            responsing: Sender<utok>,
        }

        enum RR<T: Transformer> {
            Cmd(Command),
            Ctx(SessionContext<T::Cache>),
        }

        let (rrs, rrr) = channel::<RR<T>>();
        let (enq, deq) = channel();
        let t0 = {
            let rrs = rrs.clone();
            thread::spawn(move || {
                for cmd in receiver {
                    rrs.send(RR::Cmd(cmd)).unwrap();
                }
            })
        };

        let Self {
            transformer,
            sample,
        } = self;
        let max_seq_len = transformer.model().max_position_embeddings();
        let eos = transformer.model().eos_token_id();
        let transformer = Arc::new(transformer);

        let t1 = {
            let transformer = transformer.clone();
            let enq = enq.clone();
            thread::spawn(move || {
                let mut sessions = HashMap::new();
                let mut removing = HashSet::new();
                for rr in rrr {
                    match rr {
                        RR::Cmd(Command::Infer {
                            id,
                            prompt,
                            responsing,
                        }) => {
                            let ctx = match sessions.entry(id) {
                                Entry::Occupied(ctx) => ctx.remove(),
                                Entry::Vacant(_) => SessionContext::new(&*transformer, id),
                            };
                            enq.send(Temp::<T> {
                                ctx,
                                prompts: prompt,
                                responsing,
                            })
                            .unwrap();
                        }
                        RR::Cmd(Command::Drop { id }) => {
                            if sessions.remove(&id).is_none() {
                                removing.insert(id);
                            }
                        }
                        RR::Ctx(ctx) => {
                            if !removing.remove(&ctx.0.id) {
                                sessions.insert(ctx.0.id, ctx);
                            }
                        }
                    }
                }
            })
        };
        let t2 = {
            thread::spawn(move || {
                for mut temp in deq {
                    let token = transformer.decode(
                        vec![temp.ctx.request(&temp.prompts, max_seq_len)],
                        &sample.lock().unwrap(),
                    )[0]
                    .1;
                    if token != eos {
                        temp.responsing.send(token).unwrap();
                        enq.send(Temp {
                            prompts: vec![token],
                            ..temp
                        })
                        .unwrap();
                    } else {
                        rrs.send(RR::Ctx(temp.ctx)).unwrap();
                    }
                }
            })
        };

        vec![t0, t1, t2]
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
