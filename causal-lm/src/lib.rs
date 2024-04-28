//! 提供因果语言模型的特性定义。

#![deny(warnings, missing_docs)]

mod query_context;
mod sample;

pub use query_context::QueryContext;
pub use sample::SampleArgs;

use common::{upos, utok};
use std::path::Path;
use tensor::{udim, Tensor};

/// 模型。
pub trait Model: Sized {
    /// 用于模型加载的元数据。
    type Meta;
    /// 模型加载中可能的错误。
    type Error;
    /// 从文件系统加载模型。
    fn load(model_dir: impl AsRef<Path>, meta: Self::Meta) -> Result<Self, Self::Error>;
}

/// 因果语言模型。
pub trait CausalLM: Model {
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
    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok>;
}

/// 解码的要求。
pub struct DecodingMeta {
    /// 查询的长度。
    pub num_query: usize,
    /// 解码的长度。
    pub num_decode: usize,
}

/// 解码的要求。
pub struct SampleMeta {
    /// 解码的长度。
    pub num_decode: usize,
    /// 采样参数。
    pub args: SampleArgs,
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

/// 测试模型实现。
pub fn test_impl<M>(meta: M::Meta, prompt: &[utok])
where
    M: CausalLM,
    M::Error: std::fmt::Debug,
{
    use std::time::Instant;

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let t0 = Instant::now();
    let model = M::load(model_dir, meta).unwrap();
    let t1 = Instant::now();
    println!("load {:?}", t1 - t0);

    let mut cache = model.new_cache();

    let mut prompt = prompt.to_vec();
    let mut pos = 0;

    while prompt != &[model.eos_token()] {
        let token_embedded = CausalLM::token_embed(&model, prompt.iter().copied());

        let queries = [QueryContext {
            cache: Some(&mut cache),
            range: pos..pos + prompt.len() as upos,
        }];
        let hidden_state = CausalLM::forward(&model, queries, token_embedded);

        let decoding = [DecodingMeta {
            num_query: prompt.len(),
            num_decode: 1,
        }];
        let logits = CausalLM::decode(&model, decoding, hidden_state);

        let args = [SampleMeta {
            num_decode: 1,
            args: SampleArgs::default(),
        }];
        let tokens = CausalLM::sample(&model, args, logits);

        println!("{:?}", tokens);
        pos += prompt.len() as upos;
        prompt = tokens;
    }
}
