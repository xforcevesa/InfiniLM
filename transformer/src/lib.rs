//! Common code for transformers.

#![deny(warnings, missing_docs)]

mod cache;

use common::{upos, utok};
use tensor::udim;

pub use cache::LayerCache;

/// A request to decode a sequence.
pub struct Request<'a, Storage> {
    /// Prompt of this request.
    pub prompt: Prompt<'a>,
    /// Context cache of this request.
    pub cache: &'a mut [LayerCache<Storage>],
    /// Position of `prompt` in context.
    pub pos: upos,
}

/// User prompt in transformer inference once.
pub enum Prompt<'a> {
    /// Prefill the sequence with tokens.
    Prefill(&'a [utok]),
    /// Decode the next token.
    Decode(utok),
}

impl<S> Request<'_, S> {
    /// Tokens in the prompt.
    #[inline]
    pub const fn tokens(&self) -> &[utok] {
        match &self.prompt {
            Prompt::Prefill(tokens) => tokens,
            Prompt::Decode(token) => std::slice::from_ref(&token),
        }
    }

    /// Length of tokens in the prompt.
    #[inline]
    pub const fn seq_len(&self) -> udim {
        match self.prompt {
            Prompt::Prefill(tokens) => tokens.len() as _,
            Prompt::Decode(_) => 1,
        }
    }

    /// Length of tokens in attention computation.
    #[inline]
    pub const fn att_len(&self) -> udim {
        self.pos + self.seq_len()
    }
}
