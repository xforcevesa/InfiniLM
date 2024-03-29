use crate::session::{Infer, SessionContext};
use std::sync::{Condvar, Mutex};

pub struct Task<Cache> {
    pub ctx: SessionContext<Cache>,
    pub infer: Box<Infer>,
}

pub struct Batcher<Cache> {
    queue: Mutex<Vec<Task<Cache>>>,
    condvar: Condvar,
}

impl<Cache> Batcher<Cache> {
    #[inline]
    pub fn new() -> Self {
        Self {
            queue: Default::default(),
            condvar: Default::default(),
        }
    }

    #[inline]
    pub fn enq(&self, task: Task<Cache>) {
        self.queue.lock().unwrap().push(task);
        self.condvar.notify_one();
    }

    #[inline]
    pub fn deq(&self) -> Vec<Task<Cache>> {
        std::mem::take(
            &mut *self
                .condvar
                .wait_while(self.queue.lock().unwrap(), |q| q.is_empty())
                .unwrap(),
        )
    }
}
