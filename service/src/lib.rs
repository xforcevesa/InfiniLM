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

#[derive(Debug)]
#[non_exhaustive]
pub enum Device {
    Cpu,
    NvidiaGpu(Vec<u32>),
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
                Device::NvidiaGpu(devices) => match devices.as_slice() {
                    &[] => dispatch::run(nvidia::transformer(model_dir, 0), sample, receiver),
                    &[i] => dispatch::run(nvidia::transformer(model_dir, i as _), sample, receiver),
                    dev => dispatch::run(
                        nvidia::distributed(model_dir, dev.iter().map(|&d| d as _)),
                        sample,
                        receiver,
                    ),
                },
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
    use std::{io::Write, iter::zip, time::Duration};
    use tokio::{runtime::Builder, task::JoinSet, time::sleep};

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let runtime = Builder::new_current_thread().enable_time().build().unwrap();
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
        Device::NvidiaGpu(vec![0]),
    );

    let mut set = JoinSet::new();
    let tasks = vec![
        ("Say \"Hi\" to me.", Color::Yellow),
        ("Hi", Color::Red),
        ("Where is the capital of France?", Color::Green),
    ];

    let sessions = tasks.iter().map(|_| service.launch()).collect::<Vec<_>>();

    let handle = sessions.last().unwrap().handle();
    set.spawn(async move {
        sleep(Duration::from_secs(3)).await;
        handle.abort();
    });

    for ((prompt, color), mut session) in zip(tasks, sessions) {
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

#[cfg(all(feature = "nvidia"))]
pub fn synchronize() {
    #[cfg(detected_cuda)]
    {
        use transformer_nv::cuda;
        cuda::init();
        for i in 0..cuda::Device::count() {
            cuda::Device::new(i as _)
                .retain_primary()
                .apply(|ctx| ctx.synchronize());
        }
    }
}
