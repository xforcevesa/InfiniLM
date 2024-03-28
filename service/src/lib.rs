mod cpu;
#[cfg(detected_cuda)]
mod nvidia;
mod session;
mod task;
mod template;

use common::utok;
use session::SessionComponent;
use std::{
    path::Path,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};
use task::Task;
use template::Template;
use tokenizer::{BPECommonNormalizer, Normalizer, Tokenizer, VocabTxt, BPE};
use transformer::SampleArgs;

pub use session::Session;

#[macro_use]
extern crate log;

pub struct Service {
    session_component: Arc<SessionComponent>,
    sample: Arc<Mutex<SampleArgs>>,
    _manager: JoinHandle<()>,
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
        let (sender, receiver) = channel();
        Service {
            session_component: Arc::new(SessionComponent {
                template: template(&model_dir),
                normalizer: normalizer(&model_dir),
                tokenizer: tokenizer(&model_dir),
                sender,
            }),
            sample: sample.clone(),
            _manager: thread::spawn(move || {
                match device {
                    Device::Cpu => {
                        let mut task = Task::new(cpu::transformer(model_dir), sample);
                        for cmd in receiver {
                            task.invoke(cmd);
                        }
                    }
                    #[cfg(detected_cuda)]
                    Device::NvidiaGpu(n) => {
                        let mut task = Task::new(nvidia::transformer(model_dir, n), sample);
                        for cmd in receiver {
                            task.invoke(cmd);
                        }
                    }
                    #[cfg(not(detected_cuda))]
                    _ => panic!("Unsupported device"),
                };
            }),
        }
    }

    #[inline]
    pub fn launch(&self) -> Session {
        self.session_component.clone().into()
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

enum Command {
    Infer {
        id: usize,
        prompt: Vec<utok>,
        responsing: Sender<utok>,
    },
    Drop {
        id: usize,
    },
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

    let model_dir = "../../TinyLlama-1.1B-Chat-v1.0_F16/";
    if !Path::new(model_dir).exists() {
        return;
    }

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

    let mut session = service.launch();
    let t0 = std::thread::spawn(move || {
        session.chat("Say \"Hi\" to me.", |s| {
            print!("{}", s.color(Color::Yellow));
            std::io::stdout().flush().unwrap();
        });
    });

    let mut session = service.launch();
    let t1 = std::thread::spawn(move || {
        session.chat("Hi", |s| {
            print!("{}", s.color(Color::Red));
            std::io::stdout().flush().unwrap();
        });
    });

    t0.join().unwrap();
    t1.join().unwrap();
}
