use crate::LayerCache;
use common::{upos, utok};
use tensor::{udim, Tensor};

pub struct Request<'a, Id, Storage> {
    /// Identifier of this task.
    id: Id,
    /// Prompt of this request.
    tokens: &'a [utok],
    /// Context cache of this request.
    cache: &'a mut [LayerCache<Storage>],
    /// Position of `prompt` in context.
    pos: upos,
    /// Whether to decode the output.
    decode: bool,
}

impl<'a, Id, S> Request<'a, Id, S> {
    #[inline]
    pub fn new(
        id: Id,
        tokens: &'a [utok],
        cache: &'a mut [LayerCache<S>],
        pos: upos,
        decode: bool,
    ) -> Self {
        Self {
            id,
            tokens,
            cache,
            pos,
            decode,
        }
    }
}

impl<T, U> Request<'_, T, U> {
    #[inline]
    pub fn id(self) -> T {
        self.id
    }

    #[inline]
    pub fn tokens(&self) -> &[utok] {
        self.tokens
    }

    #[inline]
    pub fn cache(&mut self, layer: usize) -> (&mut Tensor<U>, &mut Tensor<U>) {
        self.cache[layer].get()
    }

    #[inline]
    pub fn pos(&self) -> upos {
        self.pos
    }

    #[inline]
    pub const fn seq_len(&self) -> udim {
        self.tokens.len() as _
    }

    #[inline]
    pub const fn att_len(&self) -> udim {
        self.pos + self.seq_len()
    }

    #[inline]
    pub fn decode(&self) -> bool {
        self.decode
    }

    #[inline]
    pub const fn purely_decode(&self) -> bool {
        self.decode && self.tokens.len() == 1
    }
}
