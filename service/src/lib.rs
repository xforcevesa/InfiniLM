mod cpu;
#[cfg(detected_cuda)]
mod nvidia;
mod session;
mod template;

use common::utok;
use cpu::CpuTask;
use session::SessionComponent;
use std::{
    path::Path,
    sync::{
        mpsc::{channel, Sender},
        Arc,
    },
    thread::{self, JoinHandle},
};
use template::Template;
use tokenizer::{BPECommonNormalizer, Normalizer, Tokenizer, VocabTxt, BPE};
use transformer_cpu::SampleArgs;

pub use session::Session;

#[macro_use]
extern crate log;

pub struct Service {
    session_component: Arc<SessionComponent>,
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
        let (sender, receiver) = channel();
        Service {
            session_component: Arc::new(SessionComponent {
                template: template(&model_dir),
                normalizer: normalizer(&model_dir),
                tokenizer: tokenizer(&model_dir),
                sender,
            }),
            _manager: thread::spawn(move || match device {
                Device::Cpu => {
                    let mut task = CpuTask::new(model_dir, sample);
                    while let Ok(cmd) = receiver.recv() {
                        task.invoke(cmd);
                    }
                }
                #[cfg(detected_cuda)]
                Device::NvidiaGpu(n) => {
                    use nvidia::task;
                    use transformer_nvidia::cuda;

                    cuda::init();
                    let dev = cuda::Device::new(n);
                    dev.set_mempool_threshold(u64::MAX);
                    dev.context()
                        .apply(|ctx| task(model_dir, sample, receiver, ctx));
                }
                #[cfg(not(detected_cuda))]
                _ => panic!("Unsupported device"),
            }),
        }
    }

    #[inline]
    pub fn launch(&self) -> Session {
        self.session_component.clone().into()
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

struct SessionContext<Cache> {
    id: usize,
    tokens: Vec<utok>,
    cache: Vec<Cache>,
}

impl<Cache> SessionContext<Cache> {
    #[inline]
    fn new(cache: Vec<Cache>, id: usize) -> Self {
        Self {
            id,
            tokens: Vec::new(),
            cache,
        }
    }

    #[inline]
    fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> usize {
        if self.tokens.len() + tokens.len() > max_seq_len {
            let pos = self.tokens.len().min(16);
            if tokens.len() > max_seq_len / 2 {
                let tokens = &tokens[tokens.len() - max_seq_len / 2..];
                self.tokens.truncate(pos);
                self.tokens.extend_from_slice(tokens);
            } else {
                let tail_len = (self.tokens.len() - pos).min(64);
                let tail = self.tokens.len() - tail_len;
                self.tokens.copy_within(tail.., pos);
                self.tokens.truncate(pos + tail_len);
                self.tokens.extend_from_slice(tokens);
            }
            pos
        } else {
            let pos = self.tokens.len();
            self.tokens.extend_from_slice(tokens);
            pos
        }
    }
}
