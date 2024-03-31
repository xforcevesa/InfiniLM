mod batcher;
mod cpu;
mod dispatch;
#[cfg(detected_cuda)]
mod nvidia;
mod session;
mod template;

use session::SessionComponent;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use template::Template;
use tokenizer::{BPECommonNormalizer, Normalizer, Tokenizer, VocabTxt, BPE};
use tokio::{sync::mpsc::unbounded_channel, task::JoinSet};
use transformer::SampleArgs;

pub use session::Session;

#[macro_use]
extern crate log;

pub struct Service {
    session_component: Arc<SessionComponent>,
    sample: Arc<Mutex<SampleArgs>>,
    _workers: JoinSet<()>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Device {
    Cpu,
    NvidiaGpu(i32),
}

impl Service {
    pub fn load_model(path: impl AsRef<Path>, sample: SampleArgs, device: Device) -> Self {
        let model_dir = path.as_ref().to_owned();
        let sample = Arc::new(Mutex::new(sample));
        let (sender, receiver) = unbounded_channel();
        Service {
            session_component: Arc::new(SessionComponent {
                template: template(&model_dir),
                normalizer: normalizer(&model_dir),
                tokenizer: tokenizer(&model_dir),
                sender,
            }),
            sample: sample.clone(),
            _workers: match device {
                Device::Cpu => dispatch::run(cpu::transformer(model_dir), sample, receiver),
                #[cfg(detected_cuda)]
                Device::NvidiaGpu(n) => {
                    dispatch::run(nvidia::transformer(model_dir, n), sample, receiver)
                }
                #[cfg(not(detected_cuda))]
                _ => panic!("Unsupported device"),
            },
        }
    }

    #[inline]
    pub fn launch(&self) -> Session {
        Session::new(self.session_component.clone())
    }

    #[inline]
    pub fn sample_args(&self) -> SampleArgs {
        self.sample.lock().unwrap().clone()
    }

    #[inline]
    pub fn set_sample_args(&self, sample: SampleArgs) {
        *self.sample.lock().unwrap() = sample;
    }
}

fn template(model_dir: impl AsRef<Path>) -> Box<dyn Template + Send + Sync> {
    let path: String = model_dir.as_ref().display().to_string();
    let path = path.to_ascii_lowercase();
    if path.contains("tinyllama") {
        Box::new(template::ChatTinyLlama)
    } else {
        Box::new(template::ChatCPM)
    }
}

fn normalizer(model_dir: impl AsRef<Path>) -> Box<dyn Normalizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(_) => return Box::new(BPECommonNormalizer {}),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(_) => return Box::new(()),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(bpe) => return Box::new(bpe),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(voc) => return Box::new(voc),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}

#[test]
fn test() {
    use colored::{Color, Colorize};
    use std::{io::Write, path::Path};
    use tokio::{runtime::Builder, task::JoinSet};

    let model_dir = "../../TinyLlama-1.1B-Chat-v1.0_F16/";
    if !Path::new(model_dir).exists() {
        return;
    }

    let runtime = Builder::new_current_thread().build().unwrap();
    let _rt = runtime.enter();

    let service = Service::load_model(
        model_dir,
        SampleArgs {
            temperature: 0.,
            top_k: usize::MAX,
            top_p: 1.,
        },
        #[cfg(not(detected_cuda))]
        Device::Cpu,
        #[cfg(detected_cuda)]
        Device::NvidiaGpu(0),
    );

    let mut set = JoinSet::new();
    let tasks = vec![
        ("Say \"Hi\" to me.", Color::Yellow),
        ("Hi", Color::Red),
        ("Where is the capital of France?", Color::Green),
    ];

    for (prompt, color) in tasks {
        let mut session = service.launch();
        set.spawn(async move {
            session
                .chat(prompt, |s| {
                    print!("{}", s.color(color));
                    std::io::stdout().flush().unwrap();
                })
                .await;
        });
    }

    runtime.block_on(async { while set.join_next().await.is_some() {} });
    runtime.shutdown_background();
}
