use crate::{
    common::{argmax, logger_init, tokenizer},
    Template,
};
use half::f16;
use std::{
    fs::read_to_string,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use tensor::reslice;
use tokenizer::Tokenizer;
use transformer_cpu::{Llama2, Memory, Prompt, Request, Transformer};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Prompt.
    #[clap(short, long)]
    prompt: String,
    /// Tokenizer file.
    #[clap(short, long)]
    tokenizer: Option<String>,
    /// Max steps.
    #[clap(short, long)]
    step: Option<usize>,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

impl GenerateArgs {
    pub fn invoke(self) {
        logger_init(&self.log);

        let model_dir = PathBuf::from(self.model);
        let step = self.step.unwrap_or(usize::MAX);

        let time = Instant::now();
        let tokenizer = tokenizer(self.tokenizer, &model_dir);
        info!("build tokenizer ... {:?}", time.elapsed());

        if self.nvidia {
            on_nvidia_gpu(model_dir, tokenizer, self.prompt, step, usize::MAX)
        } else {
            on_host(model_dir, tokenizer, self.prompt, step)
        }
    }
}

fn on_host(
    model_dir: impl AsRef<Path>,
    tokenizer: Box<dyn Tokenizer>,
    prompt: impl AsRef<str>,
    step: usize,
) {
    let model_dir = model_dir.as_ref();
    let template: Template = if model_dir
        .as_os_str()
        .to_str()
        .unwrap()
        .to_ascii_lowercase()
        .contains("tinyllama")
    {
        Template::ChatTinyLlama
    } else {
        Template::Chat9G
    };
    let prompt = apply_template(prompt.as_ref(), template);

    let time = Instant::now();
    let model = Box::new(Memory::load_safetensors_from_dir(model_dir).unwrap());
    info!("load model ... {:?}", time.elapsed());

    let eos = model.eos_token_id();
    let time = Instant::now();
    let mut transformer = Transformer::new(model);
    let mut kv_cache = transformer.new_cache();
    info!("build transformer ... {:?}", time.elapsed());

    let time = Instant::now();
    let prompt_tokens = tokenizer.encode(&prompt);
    info!("encode prompt ... {:?}", time.elapsed());

    let time = Instant::now();
    let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
    if !tokens.is_empty() {
        transformer.decode(vec![Request {
            prompt: Prompt::Prefill(tokens),
            cache: &mut kv_cache,
            pos: 0,
        }]);
    }
    info!("prefill transformer ... {:?}", time.elapsed());

    print!("{}", prompt.replace('▁', " "));

    let mut token = *last;
    let mut pos = tokens.len();
    let time = Instant::now();
    while pos < step.min(transformer.max_seq_len()) {
        let logits = transformer.decode(vec![Request {
            prompt: transformer_cpu::Prompt::Decode(token),
            cache: &mut kv_cache,
            pos: pos as _,
        }]);
        token = argmax(reslice::<u8, f16>(logits.access().as_slice()));

        print!("{}", tokenizer.decode(token).replace('▁', " "));
        std::io::stdout().flush().unwrap();

        if token == eos {
            break;
        }

        pos += 1;
    }
    println!();
    let duration = time.elapsed();
    info!("generate ... {duration:?}");
    info!(
        "avg. speed ... {} tokens/s",
        (pos - tokens.len()) as f32 / duration.as_secs_f32()
    )
}

#[cfg(not(detected_cuda))]
fn on_nvidia_gpu(
    _: impl AsRef<Path>,
    _: Box<dyn Tokenizer>,
    _: impl AsRef<str>,
    _: usize,
    _: usize,
) {
    panic!("Nvidia GPU is not detected");
}

#[cfg(detected_cuda)]
fn on_nvidia_gpu(
    model_dir: impl AsRef<Path>,
    tokenizer: Box<dyn Tokenizer>,
    prompt: impl AsRef<str>,
    step: usize,
    preload_layers: usize,
) {
    let model_dir = model_dir.as_ref();
    let template: Template = if model_dir
        .as_os_str()
        .to_str()
        .unwrap()
        .to_ascii_lowercase()
        .contains("tinyllama")
    {
        Template::ChatTinyLlama
    } else {
        Template::Chat9G
    };
    let prompt = apply_template(prompt.as_ref(), template);

    use std::{
        fs::File,
        io::{ErrorKind::NotFound, Read},
    };
    use transformer_nvidia::{cuda, Transformer};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        panic!("No Nvidia GPU is detected");
    };

    let time = Instant::now();

    let config = File::open(model_dir.join("config.json"));
    let config = match config {
        Ok(f) => f,
        Err(e) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let safetensors = File::open(model_dir.join("model.safetensors"));
    let mut safetensors = match safetensors {
        Ok(f) => f,
        Err(e) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };
    info!("open file {:?}", time.elapsed());

    dev.set_mempool_threshold(u64::MAX);
    dev.context().apply(|ctx| {
        let time = Instant::now();
        let mut host = ctx.malloc_host::<u8>(safetensors.metadata().unwrap().len() as _);
        safetensors.read_exact(&mut host).unwrap();
        drop(safetensors);
        info!("read to host {:?}", time.elapsed());

        let compute = ctx.stream();
        let transfer = ctx.stream();

        let time = Instant::now();
        let host = Memory::load_safetensors(config, host, false).unwrap();
        let eos = host.eos_token_id();
        let mut transformer = Transformer::new(&host, preload_layers, &transfer);
        let mut kv_cache = transformer.new_cache(&compute);
        info!("build model host: {:?}", time.elapsed());

        let step = step.min(host.max_position_embeddings());
        let time = Instant::now();
        let prompt_tokens = tokenizer.encode(&prompt);
        info!("encode prompt ... {:?}", time.elapsed());

        let time = Instant::now();
        let (last, tokens) = prompt_tokens.split_last().expect("prompt is empty");
        if !tokens.is_empty() {
            transformer.update(tokens, &mut kv_cache, 0, &compute, &transfer);
        }
        info!("prefill transformer ... {:?}", time.elapsed());

        print!("{}", prompt.replace('▁', " "));

        let mut token = *last;
        let mut pos = tokens.len();
        let time = Instant::now();
        while pos < step {
            let logits = transformer.decode(token, &mut kv_cache, pos as _, &compute, &transfer);
            token = argmax(logits);

            print!("{}", tokenizer.decode(token).replace('▁', " "));
            std::io::stdout().flush().unwrap();

            if token == eos {
                break;
            }

            pos += 1;
        }
        println!();
        let duration = time.elapsed();
        info!("generate ... {duration:?}");
        info!(
            "avg. speed ... {} tokens/s",
            (pos - tokens.len()) as f32 / duration.as_secs_f32()
        )
    });
}

#[inline]
fn apply_template(prompt: &str, template: Template) -> String {
    let maybe_file = Path::new(&prompt);
    let prompt = if maybe_file.is_file() {
        read_to_string(maybe_file).unwrap()
    } else {
        prompt.to_string()
    };
    let prompt = prompt.trim();
    let mut ans = String::new();
    match template {
        Template::Chat9G => {
            ans.push_str("<s>");
            match prompt.chars().next() {
                Some(c) if c.is_ascii_alphabetic() => ans.push(' '),
                _ => {}
            }
            ans.push_str(prompt);
            ans
        }
        Template::ChatTinyLlama => {
            match prompt.chars().next() {
                Some(c) if c.is_ascii_alphabetic() => ans.push('▁'),
                _ => {}
            }
            for c in prompt.chars() {
                ans.push(match c {
                    ' ' => '▁',
                    c => c,
                });
            }
            ans
        }
    }
}
