use crate::{argmax, Command};
use common::upos;
use std::{
    collections::HashMap, fs::File, io::Read, path::Path, sync::mpsc::Receiver, time::Instant,
};
use transformer_cpu::{Llama2, Memory};
use transformer_nvidia::{
    cuda::{ContextGuard, Stream},
    LayerCache, Transformer,
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
    let max_seq_len = host.max_position_embeddings() as upos;
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
                    .or_insert_with(|| SessionContext::new(&transformer, &transfer));

                let time = Instant::now();
                let (last, tokens) = prompt.split_last().expect("prompt is empty");
                if !tokens.is_empty() {
                    transformer.update(tokens, &mut ctx.cache, ctx.pos, &compute, &transfer);
                }
                info!("prefill transformer ... {:?}", time.elapsed());

                ctx.pos += tokens.len() as upos;
                let mut token = *last;
                while ctx.pos < max_seq_len {
                    let logits =
                        transformer.decode(token, &mut ctx.cache, ctx.pos, &compute, &transfer);
                    token = argmax(logits);
                    ctx.pos += 1;
                    if token == eos {
                        break;
                    }
                    responsing.send(token).unwrap();
                }
            }
            Command::Drop { id } => {
                sessions.remove(&id);
            }
        }
    }
}

struct SessionContext<'ctx> {
    pos: upos,
    cache: Vec<LayerCache<'ctx>>,
}

impl<'ctx> SessionContext<'ctx> {
    #[inline]
    fn new(transformer: &Transformer, transfer: &'ctx Stream) -> Self {
        Self {
            pos: 0,
            cache: transformer.new_cache(transfer),
        }
    }
}
