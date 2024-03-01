use super::ServiceArgs;
use crate::common::{argmax, tokenizer};
use common::upos;
use std::{collections::HashMap, path::Path, time::Instant};
use transformer_cpu::{model_parameters::Memory, LayerCache, Transformer};

pub(super) fn run(args: ServiceArgs) {
    let model_dir = Path::new(&args.model);

    let time = Instant::now();
    let tokenizer = tokenizer(args.tokenizer, &model_dir);
    info!("build tokenizer ... {:?}", time.elapsed());

    let time = Instant::now();
    let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
    info!("load model ... {:?}", time.elapsed());

    let time = Instant::now();
    let mut transformer = Transformer::new(model);
    info!("build transformer ... {:?}", time.elapsed());

    struct SessionContext {
        pos: upos,
        kv_cache: Vec<LayerCache>,
    }

    let mut sessions = HashMap::<usize, SessionContext>::new();

    loop {
        let id = 0;
        let prompt = "The quick brown fox jumps over the lazy dog";

        let session = sessions.entry(id).or_insert_with(|| SessionContext {
            pos: 0,
            kv_cache: transformer.new_cache(),
        });

        let prompt_tokens = tokenizer.encode(&prompt.trim());
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
        if !tokens.is_empty() {
            transformer.update(tokens, &mut session.kv_cache, session.pos as _);
            session.pos += tokens.len() as upos;
        }

        let mut token = *last;
        let max_pos = transformer.max_seq_len() as upos;
        let mut out = String::new();
        while session.pos < max_pos {
            let logits = transformer.forward(token, &mut session.kv_cache, session.pos as _);
            let next = argmax(logits);

            token = next;
            session.pos += 1;

            out.push_str(&tokenizer.decode(next).replace('▁', " "));
        }
    }
}
