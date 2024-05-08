use crate::{batcher::Batcher, ServiceComponent};
use causal_lm::{CausalLM, DecodingMeta, QueryContext, SampleArgs, SampleMeta};
use common::{upos, utok};
use log::info;
use std::{
    borrow::Cow,
    cmp::Ordering::{Equal, Greater, Less},
    error, fmt,
    iter::zip,
    mem::{replace, take},
    ops::Range,
    sync::{Arc, Mutex},
};
use tensor::Tensor;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// 会话。
pub struct Session<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub sample: SampleArgs,
    cache: Option<Tensor<M::Storage>>,
    dialog: Vec<Arc<Sentence>>,
    tail: Vec<utok>,
}

impl<M: CausalLM> From<Arc<ServiceComponent<M>>> for Session<M> {
    #[inline]
    fn from(component: Arc<ServiceComponent<M>>) -> Self {
        Self {
            component,
            sample: Default::default(),
            cache: Default::default(),
            dialog: Default::default(),
            tail: Default::default(),
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
            cache: self.cache.as_ref().map(|cache| {
                self.component
                    .handle
                    .model
                    .duplicate_cache(cache, self.pos())
            }),
            dialog: self.dialog.clone(),
            tail: self.tail.clone(),
        }
    }

    /// 回滚对话到第 `dialog_pos` 个句子。
    pub fn revert(&mut self, dialog_pos: usize) -> Result<(), ChatError> {
        match dialog_pos.cmp(&self.dialog.len()) {
            Less => {
                self.tail = self.dialog[dialog_pos].head().to_vec();
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
        let pos = self.pos();
        let mut prefill = vec![];
        let mut prompt = self.dialog.is_empty() || !self.tail.is_empty();
        // 填充对话
        for s in dialog {
            let s = if prompt {
                self.component.template.apply_chat(s)
            } else {
                s.into()
            };

            let s = self.component.normalizer.encode(&s);
            let s = self.component.tokenizer.encode(&s);
            prefill.extend_from_slice(self.push_sentence(s));

            if !prompt {
                self.tail = vec![eos];
            }
            prompt = !prompt;
        }
        // 生成推理任务与会话的交互管道
        let (sender, receiver) = unbounded_channel();
        let cache = Arc::new(Mutex::new(Some(
            self.cache
                .take()
                .unwrap_or_else(|| self.component.handle.model.new_cache()),
        )));
        self.component.handle.batcher.enq(Task {
            tokens: prefill,
            pos,
            sample: self.sample.clone(),
            cache: cache.clone(),
            sender,
        });
        BusySession {
            session: self,
            receiver: Some(receiver),
            cache,
        }
    }

    #[inline]
    fn pos(&self) -> upos {
        self.dialog
            .last()
            .map_or(0, |s| s.pos + s.tokens.len() as upos)
    }

    /// 连接上一个句子的后续并构造新句子。
    fn push_sentence(&mut self, s: Vec<utok>) -> &[utok] {
        let pos = self.pos();
        let head_len = self.tail.len();
        self.tail.extend(s);
        self.dialog
            .push(Sentence::take(&mut self.tail, pos, head_len));
        &self.dialog.last().unwrap().tokens
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
    cache: Arc<Mutex<Option<Tensor<M::Storage>>>>,
}

impl<M: CausalLM> BusySession<'_, M> {
    /// 接收模型解码产生的文本。
    pub async fn decode(&mut self) -> Option<Cow<str>> {
        self.receiver.as_mut().unwrap().recv().await.map(|token| {
            // 记录 token
            self.session.tail.push(token);
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
        s.cache = self.cache.lock().unwrap().take();
        if !s.tail.is_empty() {
            // 只要忙会话收集到任何 token，就生成一个新的句子
            let answer = take(&mut s.tail);
            s.push_sentence(answer);
            // 无论忙会话为何丢弃，只要生成了新句子，就补充一个结束符
            s.tail = vec![s.component.handle.model.eos_token()];
        } else if let Some(last) = s.dialog.pop() {
            // 否则回滚句子
            s.tail = last.head().to_vec();
        }
    }
}

pub struct Generator<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    receiver: Option<UnboundedReceiver<utok>>,
    cache: Arc<Mutex<Option<Tensor<M::Storage>>>>,
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
        let (sender, receiver) = unbounded_channel();
        let cache = Arc::new(Mutex::new(Some(component.handle.model.new_cache())));
        component.handle.batcher.enq(Task {
            tokens,
            pos: 0,
            sample,
            cache: cache.clone(),
            sender,
        });
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
    pub batcher: Batcher<Task<M::Storage>>,
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
            let token_embedded = {
                let queries = tasks.iter().flat_map(|t| &t.tokens).copied();
                self.model.token_embed(queries)
            };
            // 锁定所有请求的 cache
            let hidden_state = {
                let mut queries = tasks
                    .iter()
                    .map(|t| (t, t.cache.lock().unwrap()))
                    .collect::<Vec<_>>();
                let queries = queries.iter_mut().map(|(task, lock)| QueryContext {
                    cache: lock.as_mut(),
                    range: task.range(),
                });
                self.model.forward(queries, token_embedded)
            };
            // 为每次推理启动一个任务执行解码工作
            let self_ = self.clone();
            tokio::task::spawn_blocking(move || {
                let num_decode = tasks
                    .iter()
                    .map(|t| if !t.sender.is_closed() { 1 } else { 0 })
                    .collect::<Vec<_>>();

                let decoding = zip(&tasks, &num_decode).map(|(t, num_decode)| DecodingMeta {
                    num_query: t.tokens.len(),
                    num_decode: *num_decode,
                });
                let logits = self_.model.decode(decoding, hidden_state);

                let args = zip(&tasks, &num_decode).map(|(t, num_decode)| SampleMeta {
                    num_decode: *num_decode,
                    args: t.sample.clone(),
                });
                let tokens = self_.model.sample(args, logits);

                let eos = self_.model.eos_token();
                zip(tasks, num_decode)
                    .filter(|(_, n)| *n > 0)
                    .map(|(t, _)| t)
                    .zip(tokens)
                    .filter(|(task, token)| *token != eos && task.sender.send(*token).is_ok())
                    .for_each(|(mut task, token)| {
                        task.pos += replace(&mut task.tokens, vec![token]).len() as upos;
                        self_.batcher.enq(task);
                    });
            });
        }
    }
}

/// 对话中的一个片段。
struct Sentence {
    /// 按 token 计数，句子在对话中的位置。
    pos: upos,
    /// 句子中来自上一个句子的后续 token 的数量。
    head_len: usize,
    /// 句子的 token 序列。
    tokens: Vec<utok>,
}

impl Sentence {
    /// 取走 `tokens` 以构造一个位于 `pos` 处的句子，
    /// 其中 `tokens` 的前 `head_len` token 是前一个句子的后续，回滚时需要重新连接。
    #[inline]
    pub fn take(tokens: &mut Vec<utok>, pos: upos, head_len: usize) -> Arc<Self> {
        Arc::new(Self {
            pos,
            head_len,
            tokens: take(tokens),
        })
    }

    /// 句子中来自前一句的后续部分。
    #[inline]
    pub fn head(&self) -> &[utok] {
        &self.tokens[..self.head_len]
    }
}

pub(crate) struct Task<Cache> {
    tokens: Vec<utok>,
    pos: upos,
    sample: SampleArgs,
    cache: Arc<Mutex<Option<Tensor<Cache>>>>,
    sender: UnboundedSender<utok>,
}

impl<Cache> Task<Cache> {
    #[inline]
    fn range(&self) -> Range<upos> {
        self.pos..self.pos + self.tokens.len() as upos
    }
}
