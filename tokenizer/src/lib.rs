mod bpe;
mod vocab_txt;

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

pub trait Tokenizer {
    fn bos(&self) -> utok;
    fn eos(&self) -> utok;
    fn vocab_size(&self) -> usize;
    fn max_piece_len(&self) -> usize;
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<utok>;
    fn decode(&self, token: utok) -> &str;
}

pub use bpe::BPE;
pub use vocab_txt::VocabTxt;

struct ByteDecoder([u8; 256]);

impl ByteDecoder {
    fn new() -> Self {
        let mut ans = Self([0; 256]);
        for (i, b) in ans.0.iter_mut().enumerate() {
            *b = i as _;
        }
        ans
    }

    fn decode<'a>(&'a self, piece: &'a str) -> &'a str {
        if let Some(byte) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
            let byte = u8::from_str_radix(byte, 16).unwrap();
            let byte = std::slice::from_ref(&self.0[byte as usize]);
            unsafe { std::str::from_utf8_unchecked(byte) }
        } else {
            piece
        }
    }
}
