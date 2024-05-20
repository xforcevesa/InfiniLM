use super::{batcher::Batcher, cache::Cache, task::Task};
use crate::ServiceComponent;
use causal_lm::{CausalLM, DecodingMeta, SampleArgs, SampleMeta};
use common::utok;
use std::{
    borrow::Cow,
    iter::zip,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

pub(super) struct TaskHandle<M: CausalLM> {
    receiver: Option<UnboundedReceiver<utok>>,
    cache: Arc<Mutex<Option<Cache<M::Storage>>>>,
}

impl<M: CausalLM> TaskHandle<M> {
    #[inline]
    pub fn take(&mut self) -> Cache<M::Storage> {
        // 停止响应接收
        let _ = self.receiver.take();
        // 取走 cache
        self.cache.lock().unwrap().take().unwrap()
    }
}

impl<M: CausalLM> ServiceComponent<M> {
    pub(super) fn infer(&self, sample: SampleArgs, mut cache: Cache<M::Storage>) -> TaskHandle<M> {
        let max = self.handle.model.max_seq_len() as usize;
        cache.reset_within(max / 4, max / 4 * 3);
        // 生成推理任务与会话的交互管道
        let cache = Arc::new(Mutex::new(Some(cache)));
        let (sender, receiver) = unbounded_channel();
        self.handle
            .batcher
            .enq(Task::new(cache.clone(), sample, sender));
        TaskHandle {
            receiver: Some(receiver),
            cache,
        }
    }

    pub(super) async fn decode(&self, x: &mut TaskHandle<M>) -> Option<Cow<str>> {
        x.receiver.as_mut().unwrap().recv().await.map(|token| {
            // detokenize and denormalize the token
            let ServiceComponent {
                normalizer,
                tokenizer,
                ..
            } = self;
            normalizer.decode(tokenizer.decode(token))
        })
    }
}

pub(crate) struct Dispatcher<M: CausalLM> {
    pub model: M,
    pub(super) batcher: Batcher<Task<M::Storage>>,
}

impl<M: CausalLM> From<M> for Dispatcher<M> {
    #[inline]
    fn from(model: M) -> Self {
        Self {
            model,
            batcher: Batcher::new(),
        }
    }
}

impl<M: CausalLM> Dispatcher<M> {
    /// 通过关闭任务队列通知推理线程退出。
    #[inline]
    pub fn stop(&self) {
        self.batcher.shutdown();
    }
}

impl<M> Dispatcher<M>
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
                .filter_map(|c| c.as_ref().map(Cache::query).filter(|q| !q.is_empty()))
                .flatten()
                .copied();
            let token_embedded = self.model.token_embed(queries);
            // 推理
            let queries = caches
                .iter_mut()
                .filter_map(|c| c.as_mut().map(Cache::as_ctx).filter(|q| q.seq_len() > 0));
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
