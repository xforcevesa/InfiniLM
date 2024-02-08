mod bpe;
mod vocab_txt;

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

pub trait Tokenizer {
    fn bos(&self) -> utok;
    fn eos(&self) -> utok;
    fn max_piece_len(&self) -> usize;
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<utok>;
    fn decode(&self, token: utok, next: utok) -> &str;
}

pub use bpe::BPE;
pub use vocab_txt::VocabTxt;
