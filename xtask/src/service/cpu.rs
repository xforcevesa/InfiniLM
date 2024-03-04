use super::{chat::apply_chat, ServiceParts};
use crate::{
    common::argmax,
    service::channel::{Query, ReceiveError, Response},
};
use common::upos;
use std::{collections::HashMap, time::Instant};
use transformer_cpu::{
    model_parameters::{Llama2, Memory},
    LayerCache, Prompt, Request, Transformer,
};

pub(super) fn run(
    ServiceParts {
        model_dir,
        template,
        tokenizer,
        mut channel,
    }: ServiceParts,
) {
    let time = Instant::now();
    let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
    info!("load model ... {:?}", time.elapsed());

    let _bos = model.bos_token_id();
    let eos = model.eos_token_id();

    let time = Instant::now();
    let mut transformer = Transformer::new(model);
    info!("build transformer ... {:?}", time.elapsed());

    struct SessionContext {
        pos: upos,
        kv_cache: Vec<LayerCache>,
    }

    let mut sessions = HashMap::<usize, SessionContext>::new();

    loop {
        let Query { id, prompt } = match channel.receive() {
            Ok(q) => q,
            Err(ReceiveError::NoQuery) => break,
            Err(e) => panic!("receive error: {e:?}"),
        };

        let prompt_tokens = tokenizer.encode(&apply_chat(&prompt, template));
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");

        let session = sessions.entry(id).or_insert_with(|| SessionContext {
            pos: 0,
            kv_cache: transformer.new_cache(),
        });

        if !tokens.is_empty() {
            assert!(transformer
                .decode(vec![Request {
                    prompt: Prompt::Prefill(tokens),
                    cache: &mut session.kv_cache,
                    pos: session.pos as _,
                }])
                .is_empty());
            session.pos += tokens.len() as upos;
        }

        let mut token = *last;
        let max_pos = transformer.max_seq_len() as upos;
        let mut out = String::new();
        while session.pos < max_pos {
            let logits = transformer.decode(vec![Request {
                prompt: Prompt::Decode(token),
                cache: &mut session.kv_cache,
                pos: session.pos as _,
            }]);
            token = argmax(&logits);
            if token == eos {
                break;
            }

            let text = tokenizer.decode(token).replace('▁', " ");
            out.push_str(&text);
            session.pos += 1;

            trace!("decode for {id}: {token:>5} {text:?}");
        }

        debug!("send response for {id}: {out:?}");
        channel.send(Response { id, prompt: out }).unwrap();
    }
}

#[test]
fn cpu_service() {
    use super::channel::PrefilledChannel;
    use crate::{common::tokenizer, Template};
    use std::path::PathBuf;

    let model_dir = PathBuf::from("../../TinyLlama-1.1B-Chat-v1.0_F16");
    if !model_dir.is_dir() {
        return;
    }

    const PROMPT: &str = "\
<|system|>
Your name is Bob.</s>
<|user|>
What's your name?</s>
<|assistant|>
";

    crate::common::logger_init(&Some("trace".into()));
    run(ServiceParts {
        tokenizer: tokenizer(None, &model_dir),
        model_dir,
        template: Template::ChatTinyLlama,
        channel: Box::new(PrefilledChannel::from_prompt(PROMPT.into())),
    });
}
