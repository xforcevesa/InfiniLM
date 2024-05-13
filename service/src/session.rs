use crate::{batcher::Batcher, sentence::Sentence, ServiceComponent};
use cache::Cache;
use causal_lm::{CausalLM, DecodingMeta, QueryContext, SampleArgs, SampleMeta};
use common::{upos, utok};
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
use tensor::Tensor;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// 会话。
pub struct Session<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub sample: SampleArgs,

    dialog: Vec<Arc<Sentence>>,
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
        self.dialog.len()
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
        match dialog_pos.cmp(&self.dialog.len()) {
            Less => {
                let end = &self.dialog[dialog_pos];
                let cache = self.cache.as_mut().unwrap();
                cache.cached = end.pos();
                cache.tokens.truncate(end.start());
                self.dialog.truncate(dialog_pos);
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
        let mut pos = self.dialog.last().map_or(0, |s| s.end());
        let mut cache = self
            .cache
            .take()
            .unwrap_or_else(|| Cache::new(&self.component.handle.model, vec![]));
        let mut tail = cache.query().to_vec();
        // 填充对话
        for s in dialog {
            let prompt = self.dialog.len() % 2 == 0;
            let s = if prompt {
                self.component.template.apply_chat(s)
            } else {
                s.into()
            };

            let s = self.component.normalizer.encode(&s);
            let s = self.component.tokenizer.encode(&s);

            let head_len = tail.len();
            tail.extend(s);
            let tokens = std::mem::replace(&mut tail, if prompt { vec![] } else { vec![eos] });

            let len = tail.len();
            cache.tokens.extend(&tokens);

            self.dialog.push(Sentence::new(tokens, pos, head_len));
            pos += len;
        }
        // 生成推理任务与会话的交互管道
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
        let s = &mut *self.session;
        // 停止响应接收
        let _ = self.receiver.take();
        // 回收 cache
        let mut cache = self.cache.lock().unwrap().take().unwrap();
        let end = s.dialog.last().map_or(0, |s| s.end());
        if cache.cached > end {
            // 只要忙会话收集到任何 token，就生成一个新的句子
            let tokens = cache.tokens[end..cache.cached].to_vec();
            s.dialog.push(Sentence::new(tokens, end as _, 0));
            // 无论忙会话为何丢弃，只要生成了新句子，就补充一个结束符
            cache.tokens.truncate(cache.cached);
            cache.tokens.push(s.component.handle.model.eos_token());
        } else if let Some(last) = s.dialog.pop() {
            // 否则回滚句子
            cache.cached = last.pos();
            cache.tokens.truncate(last.start());
        }
        s.cache = Some(cache);
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
        // 生成推理任务与会话的交互管道
        let cache = Arc::new(Mutex::new(Some(Cache::new(
            &component.handle.model,
            tokens,
        ))));
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
        // 丢弃 cache
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
            // 取出每个任务的查询
            let tokens = caches
                .iter()
                .map(|c| c.as_ref().map_or(vec![], |c| c.query().to_vec()))
                .collect::<Vec<_>>();
            // 词嵌入
            let token_embedded = self.model.token_embed(tokens.iter().flatten().copied());
            // 推理
            let hidden_state = self.model.forward(
                caches.iter_mut().map(|c| Cache::ctx(&mut **c)),
                token_embedded,
            );
            drop(caches);
            // 为每次推理启动一个任务执行解码工作
            let self_ = self.clone();
            tokio::task::spawn_blocking(move || {
                let num_decode = tasks
                    .iter()
                    .map(|t| if t.is_alive() { 1 } else { 0 })
                    .collect::<Vec<_>>();

                let decoding = zip(&tokens, &num_decode).map(|(t, num_decode)| DecodingMeta {
                    num_query: t.len(),
                    num_decode: *num_decode,
                });
                let logits = self_.model.decode(decoding, hidden_state);

                let args = zip(&tasks, &num_decode).map(|(t, num_decode)| SampleMeta {
                    num_decode: *num_decode,
                    args: t.sample().clone(),
                });
                let tokens = self_.model.sample(args, logits);

                let eos = self_.model.eos_token();
                zip(tasks, num_decode)
                    .filter(|(_, n)| *n > 0)
                    .map(|(t, _)| t)
                    .zip(tokens)
                    .filter(|(_, token)| *token != eos)
                    .for_each(|(mut task, token)| {
                        if task.push(token) {
                            self_.batcher.enq(task);
                        }
                    });
            });
        }
    }
}

mod task {
    use super::*;
    use std::sync::MutexGuard;

    pub(super) struct Task<Storage> {
        sample: SampleArgs,
        sender: UnboundedSender<utok>,

        cache: Arc<Mutex<Option<Cache<Storage>>>>,
    }

    impl<Storage> Task<Storage> {
        #[inline]
        pub fn new(
            cache: Arc<Mutex<Option<Cache<Storage>>>>,
            sample: SampleArgs,
            sender: UnboundedSender<utok>,
        ) -> Self {
            Self {
                sample,
                sender,
                cache,
            }
        }

        #[inline]
        pub fn sample(&self) -> &SampleArgs {
            &self.sample
        }
        #[inline]
        pub fn is_alive(&self) -> bool {
            !self.sender.is_closed()
        }
        #[inline]
        pub fn lock_cache(&self) -> MutexGuard<Option<Cache<Storage>>> {
            self.cache.lock().unwrap()
        }

        #[inline]
        pub fn push(&mut self, token: utok) -> bool {
            if self.sender.send(token).is_ok() {
                if let Some(cache) = self.cache.lock().unwrap().as_mut() {
                    cache.push(token);
                    return true;
                }
            }
            false
        }
    }
}

mod cache {
    use super::*;

    pub(super) struct Cache<Storage> {
        pos: upos,
        cache: Tensor<Storage>,
        pub tokens: Vec<utok>,
        pub cached: usize,
    }

    impl<Storage> Cache<Storage> {
        #[inline]
        pub fn new(t: &impl CausalLM<Storage = Storage>, tokens: Vec<utok>) -> Self {
            Self {
                pos: 0,
                cache: t.new_cache(),
                tokens,
                cached: 0,
            }
        }

        #[inline]
        pub fn duplicate(&self, t: &impl CausalLM<Storage = Storage>) -> Self {
            Self {
                pos: self.pos,
                cache: t.duplicate_cache(&self.cache, self.cached as _),
                tokens: self.tokens.clone(),
                cached: self.cached,
            }
        }

        /// 所有 token 中还没有加入缓存的部分就是这次的查询。
        #[inline]
        pub fn query(&self) -> &[utok] {
            &self.tokens[self.cached..]
        }

        pub fn ctx(opt: &mut Option<Self>) -> QueryContext<Storage> {
            if let Some(Cache {
                pos,
                cache,
                tokens,
                cached,
            }) = &mut *opt
            {
                QueryContext {
                    cache: Some(cache),
                    range: *pos + *cached as upos..*pos + tokens.len() as upos,
                }
            } else {
                QueryContext {
                    cache: None,
                    range: 0..0,
                }
            }
        }

        /// 将新采样的值加入缓存。
        #[inline]
        pub fn push(&mut self, token: utok) {
            self.cached = self.tokens.len();
            self.tokens.push(token);
        }
    }
}
