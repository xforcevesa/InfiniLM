use super::{channel::channel, chat::apply_chat, ServiceArgs};
use crate::{
    common::{argmax, tokenizer},
    service::channel::{Query, Response},
    Template,
};
use common::upos;
use std::{collections::HashMap, path::Path, time::Instant};
use transformer_cpu::{
    model_parameters::{Llama2, Memory},
    LayerCache, Transformer,
};

pub(super) fn run(args: ServiceArgs) {
    let template = if args.model.to_ascii_lowercase().contains("tinyllama") {
        Template::ChatTinyLlama
    } else {
        Template::Chat9G
    };
    let model_dir = Path::new(&args.model);

    let time = Instant::now();
    let tokenizer = tokenizer(args.tokenizer, &model_dir);
    info!("build tokenizer ... {:?}", time.elapsed());

    let time = Instant::now();
    let model = Box::new(Memory::load_safetensors_from_dir(&model_dir).unwrap());
    info!("load model ... {:?}", time.elapsed());

    let _bos = model.bos_token_id();
    let eos = model.eos_token_id();

    let time = Instant::now();
    let mut transformer = Transformer::new(model);
    info!("build transformer ... {:?}", time.elapsed());

    let time = Instant::now();
    let mut channel = channel(args.channel);
    info!("build channel ... {:?}", time.elapsed());

    struct SessionContext {
        pos: upos,
        kv_cache: Vec<LayerCache>,
    }

    let mut sessions = HashMap::<usize, SessionContext>::new();

    loop {
        let Query { id, prompt } = channel.receive().unwrap();
        let prompt = apply_chat(&prompt, template);

        let prompt_tokens = tokenizer.encode(&prompt.trim());
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");

        let session = sessions.entry(id).or_insert_with(|| SessionContext {
            pos: 0,
            kv_cache: transformer.new_cache(),
        });

        if !tokens.is_empty() {
            transformer.update(tokens, &mut session.kv_cache, session.pos as _);
            session.pos += tokens.len() as upos;
        }

        let mut token = *last;
        let max_pos = transformer.max_seq_len() as upos;
        let mut out = String::new();
        while session.pos < max_pos {
            let logits = transformer.forward(token, &mut session.kv_cache, session.pos as _);
            token = argmax(logits);
            if token == eos {
                break;
            }

            out.push_str(&tokenizer.decode(token).replace('▁', " "));
            session.pos += 1;
        }

        channel.send(Response { id, prompt: out }).unwrap();
    }
}
