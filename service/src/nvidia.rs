use crate::{session, Command};
use common::utok;
use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    sync::{mpsc::Receiver, Arc, Mutex},
    time::Instant,
};
use transformer_nvidia::{cuda::Device, LayerCache, Llama2, Request, SampleArgs, Transformer};

pub fn task(
    device: Device,
    model_dir: impl AsRef<Path>,
    sample: Arc<Mutex<SampleArgs>>,
    receiver: Receiver<Command>,
) {
    device.set_mempool_threshold(u64::MAX);
    let model_dir = model_dir.as_ref();

    let time = Instant::now();
    let config = File::open(model_dir.join("config.json")).unwrap();
    let safetensors = File::open(model_dir.join("model.safetensors")).unwrap();
    info!("open file {:?}", time.elapsed());

    let context = Arc::new(device.context());
    let transformer = Transformer::new(config, safetensors, usize::MAX, context.clone());
    let mut sessions = HashMap::new();

    let max_seq_len = transformer.model().max_position_embeddings();
    let eos = transformer.model().eos_token_id();
    while let Ok(cmd) = receiver.recv() {
        match cmd {
            Command::Infer {
                id,
                prompt,
                responsing,
            } => {
                let ctx = sessions
                    .entry(id)
                    .or_insert_with_key(|&id| SessionContext::new(&transformer, id));

                let t0 = Instant::now();
                let mut token = transformer.decode(
                    vec![ctx.request(&prompt, max_seq_len)],
                    &sample.lock().unwrap(),
                )[0]
                .1;
                let t1 = Instant::now();
                let mut len = 0;
                while token != eos {
                    responsing.send(token).unwrap();
                    token = transformer.decode(
                        vec![ctx.request(&[token], max_seq_len)],
                        &sample.lock().unwrap(),
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
                sessions.remove(&id);
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
    fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> Request<'_, usize> {
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
