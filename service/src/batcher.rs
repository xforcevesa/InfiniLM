use std::sync::{
    atomic::{
        AtomicBool,
        Ordering::{Acquire, Release},
    },
    Condvar, Mutex,
};

pub struct Batcher<T> {
    queue: Mutex<Vec<T>>,
    condvar: Condvar,
    alive: AtomicBool,
}

impl<T> Batcher<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            queue: Default::default(),
            condvar: Default::default(),
            alive: AtomicBool::new(true),
        }
    }

    #[inline]
    pub fn enq(&self, val: T) {
        self.queue.lock().unwrap().push(val);
        self.condvar.notify_one();
    }

    #[inline]
    pub fn deq(&self) -> Vec<T> {
        std::mem::take(
            &mut *self
                .condvar
                .wait_while(self.queue.lock().unwrap(), |q| {
                    q.is_empty() && self.alive.load(Acquire)
                })
                .unwrap(),
        )
    }

    #[inline]
    pub fn shutdown(&self) {
        self.alive.store(false, Release);
        self.queue.lock().unwrap().clear();
        self.condvar.notify_all();
    }
}
