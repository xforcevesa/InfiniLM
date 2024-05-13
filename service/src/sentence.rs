use common::utok;
use std::{ops::Deref, sync::Arc};

/// 对话中的一个片段。
pub(crate) struct Sentence {
    /// 按 token 计数，句子在对话中的位置。
    pos: usize,
    /// 句子中来自上一个句子的后续 token 的数量。
    head_len: usize,
    /// 句子的 token 序列。
    tokens: Vec<utok>,
}

impl Deref for Sentence {
    type Target = [utok];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.tokens
    }
}

impl Sentence {
    /// 取走 `tokens` 以构造一个位于 `pos` 处的句子，
    /// 其中 `tokens` 的前 `head_len` token 是前一个句子的后续，回滚时需要重新连接。
    #[inline]
    pub fn new(tokens: Vec<utok>, pos: usize, head_len: usize) -> Arc<Self> {
        Arc::new(Self {
            pos,
            head_len,
            tokens,
        })
    }
    /// 句子的起始位置。
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }
    /// 句子去除前一句的后续的起始位置。
    #[inline]
    pub fn start(&self) -> usize {
        self.pos + self.head_len
    }
    /// 句子的结束位置。
    #[inline]
    pub fn end(&self) -> usize {
        self.pos + self.tokens.len()
    }
}
