mod cache;
mod dialog;
mod task;

use crate::{batcher::Batcher, ServiceComponent};
use cache::Cache;
use causal_lm::{CausalLM, DecodingMeta, SampleArgs, SampleMeta};
use common::utok;
use dialog::Dialog;
use log::info;
use std::{
    borrow::Cow,
    cmp::Ordering::{Equal, Greater, Less},
    error, fmt,
    iter::zip,
    sync::{Arc, Mutex},
    vec,
};
use task::Task;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

/// 会话。
pub struct Session<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub sample: SampleArgs,

    dialog: Dialog,
    cache: Option<Cache<M::Storage>>,
}

impl<M: CausalLM> From<Arc<ServiceComponent<M>>> for Session<M> {
    #[inline]
    fn from(component: Arc<ServiceComponent<M>>) -> Self {
        Self {
            component,
            sample: Default::default(),

            dialog: Default::default(),
            cache: Default::default(),
        }
    }
}

impl<M: CausalLM> Session<M> {
    #[inline]
    pub fn dialog_pos(&self) -> usize {
        self.dialog.num_sentences()
    }

    /// 复制当前会话。
    pub fn fork(&self) -> Self {
        Self {
            component: self.component.clone(),
            sample: Default::default(),
            dialog: self.dialog.clone(),
            cache: self
                .cache
                .as_ref()
                .map(|cache| cache.duplicate(&self.component.handle.model)),
        }
    }

    /// 回滚对话到第 `dialog_pos` 个句子。
    pub fn revert(&mut self, dialog_pos: usize) -> Result<(), ChatError> {
        match dialog_pos.cmp(&self.dialog.num_sentences()) {
            Less => {
                let cache = self.cache.as_mut().unwrap();

                self.dialog.revert(dialog_pos);
                let cached = cache.revert(self.dialog.num_tokens());
                let last_prompt = self.dialog.last_prompt().map_or(0, |p| p.len());
                if cached < last_prompt {
                    let len = self.component.handle.model.max_seq_len() as usize;
                    let (tokens, pos) = self.dialog.window(len);
                    cache.reset_with(tokens, pos);
                }
                Ok(())
            }
            Equal => Ok(()),
            Greater => Err(ChatError),
        }
    }

    /// 用 dialog 重置会话，启动推理并返回忙会话。
    pub fn chat<'s, 'a>(
        &'s mut self,
        dialog: impl IntoIterator<Item = &'a str>,
    ) -> BusySession<'s, M> {
        let eos = self.component.handle.model.eos_token();
        let mut cache = self
            .cache
            .take()
            .unwrap_or_else(|| Cache::new(&self.component.handle.model, vec![]));
        // 填充对话
        for s in dialog {
            let prompt = self.dialog.num_sentences() % 2 == 0;

            let s = if prompt {
                self.component.template.apply_chat(s)
            } else {
                s.into()
            };
            let s = self.component.normalizer.encode(&s);
            let mut s = self.component.tokenizer.encode(&s);
            if !prompt {
                s.push(eos);
            }

            cache.extend(&s);
            self.dialog.push(s);
            // assert_eq!(cache.end(), self.dialog.last().unwrap().end());
        }
        // 生成推理任务与会话的交互管道
        let max = self.component.handle.model.max_seq_len() as usize;
        cache.reset_within(max / 4, max / 4 * 3);
        let cache = Arc::new(Mutex::new(Some(cache)));
        let (sender, receiver) = unbounded_channel();
        self.component
            .handle
            .batcher
            .enq(Task::new(cache.clone(), self.sample.clone(), sender));
        BusySession {
            session: self,
            receiver: Some(receiver),
            cache,
        }
    }

    fn restore_cache(&mut self, mut cache: Cache<M::Storage>) {
        let end = self.dialog.num_tokens();
        if cache.end() > end {
            // 无论忙会话为何丢弃，只要生成了新句子，就补充一个结束符
            cache.push(self.component.handle.model.eos_token());
            // 只要忙会话收集到任何 token，就生成一个新的句子
            self.dialog.push(cache.slice_tail(end).to_vec());
        }
        cache.cleanup();
        self.cache = Some(cache);
    }
}

/// 对话错误类型。
///
/// 目前唯一可能的对话错误是增量对话中句子位置异常。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChatError;

impl error::Error for ChatError {}
impl fmt::Display for ChatError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "chat error")
    }
}

/// 忙会话，表示会话正在处理推理任务，并可接收推理结果。
pub struct BusySession<'a, M: CausalLM> {
    session: &'a mut Session<M>,
    receiver: Option<UnboundedReceiver<utok>>,
    cache: Arc<Mutex<Option<Cache<M::Storage>>>>,
}

impl<M: CausalLM> BusySession<'_, M> {
    /// 接收模型解码产生的文本。
    pub async fn decode(&mut self) -> Option<Cow<str>> {
        self.receiver.as_mut().unwrap().recv().await.map(|token| {
            // detokenize and denormalize the token
            let ServiceComponent {
                normalizer,
                tokenizer,
                ..
            } = &*self.session.component;
            normalizer.decode(tokenizer.decode(token))
        })
    }
}

impl<M: CausalLM> Drop for BusySession<'_, M> {
    fn drop(&mut self) {
        info!("Drop busy session");
        // 停止响应接收
        let _ = self.receiver.take();
        // 取走 cache
        let cache = self.cache.lock().unwrap().take();
        // 回收 cache
        self.session.restore_cache(cache.unwrap());
    }
}

pub struct Generator<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    receiver: Option<UnboundedReceiver<utok>>,
    cache: Arc<Mutex<Option<Cache<M::Storage>>>>,
}

impl<M: CausalLM> Generator<M> {
    pub(crate) fn new(
        component: Arc<ServiceComponent<M>>,
        prompt: impl AsRef<str>,
        sample: SampleArgs,
    ) -> Self {
        let prompt = component.template.normalize(prompt.as_ref());
        let prompt = component.normalizer.encode(&prompt);
        let tokens = component.tokenizer.encode(&prompt);
        let mut cache = Cache::new(&component.handle.model, tokens);
        let max = component.handle.model.max_seq_len() as usize;
        cache.reset_within(max / 4, max / 4 * 3);
        // 生成推理任务与会话的交互管道
        let cache = Arc::new(Mutex::new(Some(cache)));
        let (sender, receiver) = unbounded_channel();
        component
            .handle
            .batcher
            .enq(Task::new(cache.clone(), sample, sender));
        Self {
            component,
            receiver: Some(receiver),
            cache,
        }
    }

    pub async fn decode(&mut self) -> Option<Cow<str>> {
        self.receiver.as_mut().unwrap().recv().await.map(|token| {
            // detokenize and denormalize the token
            let ServiceComponent {
                normalizer,
                tokenizer,
                ..
            } = &*self.component;
            normalizer.decode(tokenizer.decode(token))
        })
    }
}

impl<M: CausalLM> Drop for Generator<M> {
    fn drop(&mut self) {
        // 停止响应接收
        let _ = self.receiver.take();
        // 取走 cache
        let _ = self.cache.lock().unwrap().take();
    }
}

pub(crate) struct HandleComponent<M: CausalLM> {
    model: M,
    batcher: Batcher<Task<M::Storage>>,
}

impl<M: CausalLM> From<M> for HandleComponent<M> {
    #[inline]
    fn from(model: M) -> Self {
        Self {
            model,
            batcher: Batcher::new(),
        }
    }
}

impl<M: CausalLM> HandleComponent<M> {
    /// 通过关闭任务队列通知推理线程退出。
    #[inline]
    pub fn stop(&self) {
        self.batcher.shutdown();
    }
}

impl<M> HandleComponent<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    pub fn run(self: Arc<Self>) {
        while let Some(tasks) = Some(self.batcher.deq()).filter(|t| !t.is_empty()) {
            // 锁定所有请求的缓存
            let mut caches = tasks.iter().map(Task::lock_cache).collect::<Vec<_>>();
            // 统计每个任务的查询长度
            let num_query = caches
                .iter()
                .map(|c| c.as_ref().map_or(0, |c| c.query().len()))
                .collect::<Vec<_>>();
            if num_query.iter().all(|&n| n == 0) {
                continue;
            }
            // 词嵌入
            let queries = caches
                .iter()
                .filter_map(|c| c.as_ref().map(Cache::query))
                .flatten()
                .copied();
            let token_embedded = self.model.token_embed(queries);
            // 推理
            let queries = caches
                .iter_mut()
                .filter_map(|c| c.as_mut().map(Cache::as_ctx));
            let hidden_state = self.model.forward(queries, token_embedded);
            drop(caches);
            // 为每次推理启动一个任务执行解码工作
            let self_ = self.clone();
            tokio::task::spawn_blocking(move || {
                let num_decode = tasks
                    .iter()
                    .map(|t| if t.is_alive() { 1 } else { 0 })
                    .collect::<Vec<_>>();

                let decoding =
                    zip(&num_query, &num_decode).map(|(&num_query, &num_decode)| DecodingMeta {
                        num_query,
                        num_decode,
                    });
                let logits = self_.model.decode(decoding, hidden_state);

                let args = zip(&tasks, &num_decode).map(|(t, &num_decode)| SampleMeta {
                    num_decode,
                    args: t.sample().clone(),
                });
                let tokens = self_.model.sample(args, logits);

                let eos = self_.model.eos_token();
                let max = self_.model.max_seq_len() as usize;
                let min = max / 4;
                zip(tasks, num_decode)
                    .filter(|(_, n)| *n > 0)
                    .map(|(t, _)| t)
                    .zip(tokens)
                    .filter(|(_, token)| *token != eos)
                    .for_each(|(mut task, token)| {
                        if task.push(token, min, max) {
                            self_.batcher.enq(task);
                        }
                    });
            });
        }
    }
}
