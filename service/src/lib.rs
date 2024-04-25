mod batcher;
mod template;

use batcher::Batcher;
use causal_lm::{CausalLM, DecodingMeta, QueryContext, SampleArgs, SampleMeta};
use common::{upos, utok};
use core::fmt;
use std::{
    borrow::Cow,
    error,
    iter::zip,
    mem::{replace, take},
    ops::Range,
    path::Path,
    sync::{Arc, Mutex},
};
use template::Template;
use tensor::Tensor;
use tokenizer::{BPECommonNormalizer, Normalizer, Tokenizer, VocabTxt, BPE};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// 对话服务。
#[repr(transparent)]
pub struct Service<M: CausalLM>(Arc<ServiceComponent<M>>);
/// 会话。
pub struct Session<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub sample: SampleArgs,
    cache: Option<Tensor<M::Storage>>,
    dialog: Vec<Arc<Sentence>>,
    tail: Vec<utok>,
}
/// 忙会话，表示会话正在处理推理任务，并可接收推理结果。
pub struct BusySession<'a, M: CausalLM> {
    session: &'a mut Session<M>,
    receiver: Option<UnboundedReceiver<utok>>,
    cache: Arc<Mutex<Option<Tensor<M::Storage>>>>,
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

/// 服务中不变的组件，将在所有会话之间共享。
///
/// 推理线程的生命周期与这个组件绑定。
struct ServiceComponent<M: CausalLM> {
    handle: Arc<HandleComponent<M>>,
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    normalizer: Box<dyn Normalizer + Send + Sync>,
    template: Box<dyn template::Template + Send + Sync>,
}

impl<M: CausalLM> Drop for ServiceComponent<M> {
    #[inline]
    fn drop(&mut self) {
        self.handle.batcher.shutdown();
    }
}

struct HandleComponent<M: CausalLM> {
    model: M,
    batcher: Batcher<Task<M::Storage>>,
}

struct Task<Cache> {
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

impl<M> Service<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send + Sync + 'static,
{
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        let handle = Arc::new(HandleComponent {
            model: M::load(&model_dir),
            batcher: Batcher::new(),
        });
        {
            let handle = handle.clone();
            tokio::task::spawn_blocking(move || {
                // 这个线程的生命周期不小于服务的生命周期，不占用线程池
                while let Some(tasks) = Some(handle.batcher.deq()).filter(|t| !t.is_empty()) {
                    let token_embedded = {
                        let queries = tasks.iter().flat_map(|t| &t.tokens).copied();
                        handle.model.token_embed(queries)
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
                        handle.model.forward(queries, token_embedded)
                    };
                    // 为每次推理启动一个任务执行解码工作
                    let handle = handle.clone();
                    tokio::task::spawn_blocking(move || {
                        let num_decode = tasks
                            .iter()
                            .map(|t| if !t.sender.is_closed() { 1 } else { 0 })
                            .collect::<Vec<_>>();

                        let decoding =
                            zip(&tasks, &num_decode).map(|(t, num_decode)| DecodingMeta {
                                num_query: t.tokens.len(),
                                num_decode: *num_decode,
                            });
                        let logits = handle.model.decode(decoding, hidden_state);

                        let args = zip(&tasks, &num_decode).map(|(t, num_decode)| SampleMeta {
                            num_decode: *num_decode,
                            args: t.sample.clone(),
                        });
                        let tokens = handle.model.sample(args, logits);

                        let eos = handle.model.eos_token();
                        zip(tasks, num_decode)
                            .filter(|(_, n)| *n > 0)
                            .map(|(t, _)| t)
                            .zip(tokens)
                            .filter(|(task, token)| {
                                *token != eos && task.sender.send(*token).is_ok()
                            })
                            .for_each(|(mut task, token)| {
                                task.pos += replace(&mut task.tokens, vec![token]).len() as upos;
                                handle.batcher.enq(task);
                            });
                    });
                }
            });
        }
        Self(Arc::new(ServiceComponent {
            handle,
            tokenizer: tokenizer(&model_dir),
            normalizer: normalizer(&model_dir),
            template: template(model_dir),
        }))
    }
}

impl<M: CausalLM> Service<M> {
    /// 从对话服务启动一个会话。
    pub fn launch(&self) -> Session<M> {
        Session {
            component: self.0.clone(),
            sample: SampleArgs::default(),
            cache: None,
            dialog: vec![],
            tail: vec![],
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

    /// 用 dialog 重置会话，启动推理并返回忙会话。
    pub fn reset<'s, 'a>(
        &'s mut self,
        dialog: impl IntoIterator<Item = &'a str>,
    ) -> BusySession<'s, M> {
        // 重置会话状态
        self.dialog.clear();
        self.tail.clear();
        // 填充对话
        let eos = self.component.handle.model.eos_token();
        let mut prompt = true;
        let mut prefill = vec![];
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
        self.infer(prefill, 0)
    }

    /// 向对话的 `dialog_pos` 处填充 `prompt`，启动推理并返回忙会话。
    ///
    /// 如果 `dialog_pos` 位置之前有未知的句子，返回 `ChatError`。
    pub fn chat(&mut self, dialog_pos: usize, prompt: &str) -> Result<BusySession<M>, ChatError> {
        if dialog_pos > self.dialog.len() {
            Err(ChatError)
        } else {
            // tokenize and normalize the prompt
            let prompt = self.component.template.apply_chat(prompt);
            let prompt = self.component.normalizer.encode(&prompt);
            let prompt = self.component.tokenizer.encode(&prompt);
            // dialog_pos 是历经的位置，需要回滚对话
            if let Some(sentence) = self.dialog.get(dialog_pos) {
                let tail = sentence.head();
                self.tail = Vec::with_capacity(tail.len() + prompt.len());
                self.tail.extend(tail);
                self.dialog.truncate(dialog_pos);
            }
            let pos = self.pos();
            let prompt = self.push_sentence(prompt).to_vec();
            Ok(self.infer(prompt, pos))
        }
    }

    fn infer(&mut self, tokens: Vec<utok>, pos: upos) -> BusySession<M> {
        // 生成推理任务与会话的交互管道
        let (sender, receiver) = unbounded_channel();
        let cache = Arc::new(Mutex::new(Some(
            self.cache
                .take()
                .unwrap_or_else(|| self.component.handle.model.new_cache()),
        )));
        self.component.handle.batcher.enq(Task {
            tokens,
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

#[test]
fn test() {
    use colored::{Color, Colorize};
    use std::{io::Write, iter::zip};
    use tokio::{runtime::Builder, task::JoinSet};

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let runtime = Builder::new_current_thread().build().unwrap();
    let _rt = runtime.enter();

    let service = Service::<transformer_cpu::Transformer>::new(model_dir);

    let mut set = JoinSet::new();
    let tasks = vec![
        ("Say \"Hi\" to me.", Color::Yellow),
        ("Hi", Color::Red),
        ("Where is the capital of France?", Color::Green),
    ];

    let sessions = tasks.iter().map(|_| service.launch()).collect::<Vec<_>>();

    for ((prompt, color), mut session) in zip(tasks, sessions) {
        set.spawn(async move {
            let mut busy = session.chat(0, prompt).unwrap();
            while let Some(s) = busy.decode().await {
                print!("{}", s.color(color));
                std::io::stdout().flush().unwrap();
            }
        });
    }

    runtime.block_on(async { while set.join_next().await.is_some() {} });
    runtime.shutdown_background();
}

fn template(model_dir: impl AsRef<Path>) -> Box<dyn Template + Send + Sync> {
    let path: String = model_dir.as_ref().display().to_string();
    let path = path.to_ascii_lowercase();
    if path.contains("tinyllama") {
        Box::new(template::ChatTinyLlama)
    } else {
        Box::new(template::ChatCPM)
    }
}

fn normalizer(model_dir: impl AsRef<Path>) -> Box<dyn Normalizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(_) => return Box::new(BPECommonNormalizer {}),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(_) => return Box::new(()),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(bpe) => return Box::new(bpe),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(voc) => return Box::new(voc),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}
