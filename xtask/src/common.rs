use common::utok;
use log::LevelFilter;
use simple_logger::SimpleLogger;
use std::{io::ErrorKind::NotFound, path::Path};
use tokenizer::{Tokenizer, VocabTxt, BPE};

pub(crate) fn logger_init(log_level: &Option<String>) {
    let log = log_level
        .as_ref()
        .and_then(|log| match log.to_lowercase().as_str() {
            "off" | "none" => Some(LevelFilter::Off),
            "trace" => Some(LevelFilter::Trace),
            "debug" => Some(LevelFilter::Debug),
            "info" => Some(LevelFilter::Info),
            "error" => Some(LevelFilter::Error),
            _ => None,
        })
        .unwrap_or(LevelFilter::Warn);
    SimpleLogger::new().with_level(log).init().unwrap();
}

pub(crate) fn tokenizer(path: Option<String>, model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer> {
    match path {
        Some(path) => match Path::new(&path).extension() {
            Some(ext) if ext == "txt" => Box::new(VocabTxt::from_txt_file(path).unwrap()),
            Some(ext) if ext == "model" => Box::new(BPE::from_model_file(path).unwrap()),
            _ => panic!("Tokenizer file {path:?} not supported"),
        },
        None => {
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
    }
}

pub(crate) fn argmax<T: PartialOrd>(logits: &[T]) -> utok {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as _
}
