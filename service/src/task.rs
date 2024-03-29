use crate::{session, Command};
use common::utok;
use std::{
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Condvar, Mutex,
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

struct Temp<Cache> {
    ctx: SessionContext<Cache>,
    prompts: Vec<utok>,
    responsing: Sender<utok>,
}

struct Batcher<Cache> {
    queue: Mutex<VecDeque<Temp<Cache>>>,
    condvar: Condvar,
}

impl<T> Task<T>
where
    T: Transformer + Send + Sync + 'static,
    T::Cache: Send + 'static,
{
    pub fn run(self, commands: Receiver<Command>) -> Vec<JoinHandle<()>> {
        enum RR<T: Transformer> {
            Cmd(Command),
            Ctx(SessionContext<T::Cache>),
        }

        let (sender, receiver) = channel::<RR<T>>();
        let t0 = {
            let sender = sender.clone();
            thread::spawn(move || {
                for cmd in commands {
                    sender.send(RR::Cmd(cmd)).unwrap();
                }
            })
        };

        let batcher = Arc::new(Batcher::<T::Cache> {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
        });
        let Self {
            transformer,
            sample,
        } = self;
        let max_seq_len = transformer.model().max_position_embeddings();
        let eos = transformer.model().eos_token_id();
        let transformer = Arc::new(transformer);

        let t1 = {
            let transformer = transformer.clone();
            let batcher = batcher.clone();
            thread::spawn(move || {
                let mut sessions = HashMap::new();
                let mut removing = HashSet::new();
                for rr in receiver {
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
                            batcher.queue.lock().unwrap().push_back(Temp {
                                ctx,
                                prompts: prompt,
                                responsing,
                            });
                            batcher.condvar.notify_one();
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
            thread::spawn(move || loop {
                let mut queue = batcher
                    .condvar
                    .wait_while(batcher.queue.lock().unwrap(), |q| q.is_empty())
                    .unwrap();
                let mut temps = std::iter::from_fn(|| queue.pop_front()).collect::<Vec<_>>();
                drop(queue);

                let requests = temps
                    .iter_mut()
                    .map(|temp| temp.ctx.request(&temp.prompts, max_seq_len))
                    .collect::<Vec<_>>();

                let tokens = transformer
                    .decode(requests, &sample.lock().unwrap())
                    .into_iter()
                    .collect::<HashMap<_, _>>();

                let mut queue = batcher.queue.lock().unwrap();
                for temp in temps {
                    match tokens.get(&temp.ctx.0.id) {
                        Some(&token) => {
                            if token != eos {
                                temp.responsing.send(token).unwrap();
                                queue.push_back(Temp {
                                    prompts: vec![token],
                                    ..temp
                                });
                            } else {
                                sender.send(RR::Ctx(temp.ctx)).unwrap();
                            }
                        }
                        None => todo!(),
                    };
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
