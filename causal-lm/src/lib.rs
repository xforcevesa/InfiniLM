//! 提供因果语言模型的特性定义。

#![deny(warnings, missing_docs)]

mod query_context;
mod sample;

pub use query_context::QueryContext;
pub use sample::SampleArgs;

use common::{upos, utok};
use tensor::{udim, Tensor};

/// 因果语言模型。
pub trait CausalLM {
    /// 存储中间结果的类型。
    type Storage;
    /// 模型定义的句子结束符。
    fn eos_token(&self) -> utok;
    /// 创建一个新的缓存（`num_layers x 2 x num_kv_head x max_seq_len x head_dim`）。
    fn new_cache(&self) -> Tensor<Self::Storage>;
    /// 复制一个有效长度为 `pos` 的缓存。
    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage>;
    /// 对所有词执行词嵌入（`num_tokens x hidden_size`）。
    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage>;
    /// 对词嵌入张量执行 Transformer 计算（`num_tokens x hidden_size`）。
    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a;
    /// 对词嵌入张量执行解码计算（`num_decoding_tokens` x `vocab_size`）。
    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>;
    /// 对 logits 进行采样。
    fn sample(&self, logits: Tensor<Self::Storage>, args: SampleArgs) -> Vec<utok>;
}

/// 解码的要求。
pub struct DecodingMeta {
    /// 查询的长度。
    pub num_query: usize,
    /// 解码的长度。
    pub num_decode: usize,
}

/// 生成位置张量。
#[inline]
pub fn pos<'a, S: 'a>(
    queries: impl IntoIterator<Item = &'a QueryContext<'a, S>>,
    nt_hint: udim,
) -> Tensor<Vec<upos>> {
    let mut ans = Vec::with_capacity(nt_hint as usize);
    for query in queries {
        ans.extend(query.pos()..query.att_len());
    }
    Tensor::new(tensor::DataType::U32, &[ans.len() as _], ans)
}
