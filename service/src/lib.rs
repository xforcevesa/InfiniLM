mod cpu;
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
    time::Instant,
};
use template::Template;
use tokenizer::{Tokenizer, VocabTxt, BPE};

pub use session::Session;

#[macro_use]
extern crate log;

pub struct Service {
    session_component: Arc<SessionComponent>,
    _manager: JoinHandle<()>,
}

impl Service {
    pub fn load_model(path: impl AsRef<Path>) -> Self {
        let model_dir = path.as_ref().to_owned();

        let template: Box<dyn Template> = {
            let path: String = model_dir.display().to_string();
            let path = path.to_ascii_lowercase();
            if path.contains("tinyllama") {
                Box::new(template::ChatTinyLlama)
            } else {
                Box::new(template::ChatCPM)
            }
        };

        let time = Instant::now();
        let tokenizer = tokenizer(&model_dir);
        info!("build tokenizer ... {:?}", time.elapsed());

        let (sender, receiver) = channel();
        Service {
            session_component: Arc::new(SessionComponent {
                template,
                tokenizer,
                sender,
            }),
            _manager: thread::spawn(move || {
                let mut task = CpuTask::new(model_dir);
                while let Ok(cmd) = receiver.recv() {
                    task.invoke(cmd);
                }
                // {
                //     use transformer_nvidia::cuda;

                //     cuda::init();
                //     let Some(dev) = cuda::Device::fetch() else {
                //         panic!("No Nvidia GPU is detected");
                //     };

                //     dev.set_mempool_threshold(u64::MAX);
                // }
            }),
        }
    }

    #[inline]
    pub fn launch(&self) -> Session {
        self.session_component.clone().into()
    }
}

enum Command {
    Chat {
        id: usize,
        prompt: Vec<utok>,
        responsing: Sender<utok>,
    },
    Drop {
        id: usize,
    },
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer> {
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
