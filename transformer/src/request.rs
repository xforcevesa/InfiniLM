use crate::LayerCache;
use common::{upos, utok};
use tensor::udim;

pub struct Request<'a, Id, Storage> {
    /// Identifier of this task.
    pub id: Id,
    /// Prompt of this request.
    pub tokens: &'a [utok],
    /// Context cache of this request.
    pub cache: &'a mut [LayerCache<Storage>],
    /// Position of `prompt` in context.
    pub pos: upos,
}

impl<T, U> Request<'_, T, U> {
    #[inline]
    pub const fn seq_len(&self) -> udim {
        self.tokens.len() as _
    }

    #[inline]
    pub const fn att_len(&self) -> udim {
        self.pos + self.seq_len()
    }
}
