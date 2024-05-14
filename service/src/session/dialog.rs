use common::utok;
use std::sync::Arc;

#[derive(Clone, Default, Debug)]
pub(crate) struct Dialog(Vec<Arc<(Vec<utok>, usize)>>);

impl Dialog {
    #[inline]
    pub fn num_sentences(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.0.last().map_or(0, |s| s.1)
    }

    #[inline]
    pub fn revert(&mut self, len: usize) {
        self.0.truncate(len);
    }

    #[inline]
    pub fn last_prompt(&self) -> Option<&[utok]> {
        self.0
            .last()
            .filter(|_| self.0.len() % 2 != 0)
            .map(|s| &*s.0)
    }

    #[inline]
    pub fn push(&mut self, tokens: Vec<utok>) {
        let len = self.num_tokens() + tokens.len();
        self.0.push(Arc::new((tokens, len)))
    }

    #[inline]
    pub fn window(&self, len: usize) -> (Vec<utok>, usize) {
        let start = self.num_tokens().saturating_sub(len);
        let mut iter = self.0.iter().map(|s| &*s.0);
        let mut pos = 0;
        for tokens in iter.by_ref() {
            if let Some(len) = start.checked_sub(pos) {
                let ans = tokens[len..]
                    .iter()
                    .chain(iter.flatten())
                    .copied()
                    .collect();
                return (ans, pos + len);
            } else {
                pos += tokens.len();
            }
        }
        unreachable!()
    }
}
