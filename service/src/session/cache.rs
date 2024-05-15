use causal_lm::{CausalLM, QueryContext};
use common::{upos, utok};
use std::ops::Range;
use tensor::Tensor;

pub(super) struct Cache<Storage> {
    /// 可映射的 token 序列。
    tokens: Vec<utok>,
    /// token 序列在整个对话中的位置。
    pos: usize,
    /// 缓存在 token 序列中的范围。
    cached: Range<usize>,
    /// 计算缓存。
    cache: Tensor<Storage>,
}

impl<Storage> Cache<Storage> {
    /// 生成一个空白的缓存结构，准备填充 `tokens`。
    #[inline]
    pub fn new(t: &impl CausalLM<Storage = Storage>, tokens: Vec<utok>) -> Self {
        Self {
            tokens,
            pos: 0,
            cached: 0..0,
            cache: t.new_cache(),
        }
    }
    /// 复制缓存结构。
    #[inline]
    pub fn duplicate(&self, t: &impl CausalLM<Storage = Storage>) -> Self {
        assert_eq!(self.cached.start, 0);
        Self {
            tokens: self.tokens.clone(),
            pos: self.pos,
            cached: self.cached.clone(),
            cache: t.duplicate_cache(&self.cache, self.cached.end as _),
        }
    }
    /// 回滚缓存到 `pos`，并返回剩余的有效缓存长度。
    pub fn revert(&mut self, pos: usize) -> usize {
        // 只能在闲时回滚，因此 cache 和 tokens 起始位置对齐
        assert_eq!(self.cached.start, 0);
        // 回滚之后，tokens.len()、cached.end、pos 不能大于新的 pos
        let len = pos.saturating_sub(self.pos);
        // 1. tokens.len() 不大于 pos；
        self.tokens.truncate(len);
        // 2. cached.end 不大于 pos；
        self.cached.end = self.cached.end.min(len);
        // 3. pos 不大于 pos；
        self.pos = self.pos.min(pos);
        // 返回当前的缓存长度
        self.cached.len()
    }
    /// 扩展待填充 token。
    #[inline]
    pub fn extend(&mut self, tokens: &[utok]) {
        self.tokens.extend_from_slice(tokens);
    }
    /// 所有 token 中还没有加入缓存的部分就是这次的查询。
    #[inline]
    pub fn query(&self) -> &[utok] {
        &self.tokens[self.cached.end..]
    }
    /// 生成对应的查询上下文。
    #[inline]
    pub fn as_ctx(&mut self) -> QueryContext<Storage> {
        let Cache {
            pos: _pos,
            cache,
            tokens,
            cached,
        } = self;
        QueryContext {
            cache: Some(cache),
            range: cached.len() as upos..(tokens.len() - cached.start) as upos,
        }
    }

    /// 将新采样的值加入缓存。
    #[inline]
    pub fn push(&mut self, token: utok) {
        self.cached.end = self.tokens.len();
        self.tokens.push(token);
    }
    /// 已采样的最后一个词在对话中的位置。
    #[inline]
    pub fn end(&self) -> usize {
        self.pos + self.tokens.len()
    }
    /// 提取尾部词序列。
    #[inline]
    pub fn slice_tail(&self, pos: usize) -> &[utok] {
        let known = pos.checked_sub(self.pos).unwrap();
        &self.tokens[known..]
    }

    /// 重置缓存窗口。
    pub fn reset_within(&mut self, min: usize, max: usize) {
        if self.tokens.len() - self.cached.start >= max {
            self.cached.start = self.tokens.len() - min;
            self.cached.end = self.cached.start;
        }
    }
    /// 重置缓存窗口。
    pub fn reset_with(&mut self, tokens: Vec<utok>, pos: usize) {
        self.tokens = tokens;
        self.pos = pos;
        self.cached = 0..0;
    }
    /// 清理缓存中已脱离缓存窗口的部分。
    pub fn cleanup(&mut self) {
        let to_remove = self.cached.start;
        if to_remove > 0 {
            self.tokens.copy_within(to_remove.., 0);
            self.pos += to_remove;
            self.tokens.truncate(self.tokens.len() - to_remove);
            self.cached.start = 0;
            self.cached.end -= to_remove;
        }
    }
}
