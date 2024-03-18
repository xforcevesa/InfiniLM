use crate::Command;
use common::utok;
use std::{
    collections::HashMap, fs::File, io::Read, path::Path, sync::mpsc::Receiver, time::Instant,
};
use transformer_cpu::{Llama2, Memory, SampleArgs};
use transformer_nvidia::{
    cuda::{ContextGuard, Stream},
    LayerCache, Request, Transformer,
};

pub fn task(model_dir: impl AsRef<Path>, receiver: Receiver<Command>, ctx: &ContextGuard) {
    let model_dir = model_dir.as_ref();

    let time = Instant::now();
    let config = File::open(model_dir.join("config.json")).unwrap();
    let mut safetensors = File::open(model_dir.join("model.safetensors")).unwrap();
    info!("open file {:?}", time.elapsed());

    let time = Instant::now();
    let mut host = ctx.malloc_host::<u8>(safetensors.metadata().unwrap().len() as _);
    safetensors.read_exact(&mut host).unwrap();
    drop(safetensors);
    info!("read to host {:?}", time.elapsed());

    let compute = ctx.stream();
    let transfer = ctx.stream();

    let time = Instant::now();
    let host = Memory::load_safetensors(config, host, false).unwrap();
    let max_seq_len = host.max_position_embeddings();
    let eos = host.eos_token_id();
    let mut transformer = Transformer::new(&host, usize::MAX, &transfer);
    info!("build model host: {:?}", time.elapsed());

    let mut sessions = HashMap::new();

    while let Ok(cmd) = receiver.recv() {
        match cmd {
            Command::Infer {
                id,
                prompt,
                responsing,
            } => {
                let ctx = sessions
                    .entry(id)
                    .or_insert_with_key(|&id| SessionContext::new(&transformer, id, &transfer));

                let time = Instant::now();
                let mut token = transformer.decode(
                    vec![ctx.request(&prompt, max_seq_len)],
                    SampleArgs::Top,
                    &compute,
                    &transfer,
                )[0]
                .1;
                info!("prefill transformer ... {:?}", time.elapsed());

                while token != eos {
                    responsing.send(token).unwrap();
                    token = transformer.decode(
                        vec![ctx.request(&[token], max_seq_len)],
                        SampleArgs::Top,
                        &compute,
                        &transfer,
                    )[0]
                    .1;
                }
            }
            Command::Drop { id } => {
                sessions.remove(&id);
            }
        }
    }
}

struct SessionContext<'a>(super::SessionContext<LayerCache<'a>>);

impl<'a> SessionContext<'a> {
    #[inline]
    fn new(transformer: &Transformer, id: usize, stream: &'a Stream) -> Self {
        Self(super::SessionContext::new(
            transformer.new_cache(stream),
            id,
        ))
    }

    #[inline]
    fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> Request<'_, 'a, usize> {
        let pos = self.0.request(tokens, max_seq_len);
        Request {
            id: self.0.id,
            tokens: &self.0.tokens[pos..],
            cache: &mut self.0.cache,
            pos: pos as _,
        }
    }
}
