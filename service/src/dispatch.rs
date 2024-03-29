use crate::{
    batcher::{Batcher, Task},
    session::{Message, SessionContext},
};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::{
        mpsc::{channel, Receiver},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};
use transformer::{SampleArgs, Transformer};

pub fn run<T>(
    transformer: T,
    sample: Arc<Mutex<SampleArgs>>,
    messages: Receiver<Message>,
) -> Vec<JoinHandle<()>>
where
    T: Transformer + Send + Sync + 'static,
    T::Cache: Send + 'static,
{
    enum X<Cache> {
        Msg(Message),
        Ctx(SessionContext<Cache>),
    }

    let (sender, receiver) = channel();
    let t0 = {
        let sender = sender.clone();
        thread::spawn(move || {
            for msg in messages {
                sender.send(X::Msg(msg)).unwrap();
            }
        })
    };

    let max_seq_len = transformer.model().max_position_embeddings();
    let eos = transformer.model().eos_token_id();
    let transformer = Arc::new(transformer);
    let batcher = Arc::new(Batcher::new());

    let t1 = {
        let transformer = transformer.clone();
        let batcher = batcher.clone();
        thread::spawn(move || {
            let mut sessions = HashMap::new();
            let mut removing = HashSet::new();
            for x in receiver {
                match x {
                    X::Msg(Message::Infer(id, infer)) => {
                        let ctx = match sessions.entry(id) {
                            Entry::Occupied(ctx) => ctx.remove(),
                            Entry::Vacant(_) => SessionContext::new(transformer.new_cache(), id),
                        };
                        batcher.enq(Task { ctx, infer });
                    }
                    X::Msg(Message::Drop(id)) => {
                        if sessions.remove(&id).is_none() {
                            removing.insert(id);
                        }
                    }
                    X::Ctx(ctx) => {
                        if !removing.remove(&ctx.id) {
                            sessions.insert(ctx.id, ctx);
                        }
                    }
                }
            }
        })
    };
    let t2 = {
        thread::spawn(move || loop {
            let mut tasks = batcher.deq();

            let requests = tasks
                .iter_mut()
                .map(|task| task.ctx.request(&task.infer.prompt, max_seq_len))
                .collect::<Vec<_>>();

            let tokens = transformer
                .decode(requests, &sample.lock().unwrap())
                .into_iter()
                .collect::<HashMap<_, _>>();

            for mut task in tasks {
                match tokens.get(&task.ctx.id) {
                    Some(&token) => {
                        if token != eos {
                            task.infer.responsing.send(token).unwrap();
                            task.infer.prompt = vec![token];
                            batcher.enq(task);
                        } else {
                            sender.send(X::Ctx(task.ctx)).unwrap();
                        }
                    }
                    None => todo!(),
                };
            }
        })
    };

    vec![t0, t1, t2]
}
