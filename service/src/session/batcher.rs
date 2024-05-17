use std::sync::{Condvar, Mutex};

pub struct Batcher<T> {
    queue: Mutex<(Vec<T>, bool)>,
    condvar: Condvar,
}

impl<T> Batcher<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            queue: Mutex::new((Vec::new(), true)),
            condvar: Default::default(),
        }
    }

    #[inline]
    pub fn enq(&self, val: T) {
        let mut lock = self.queue.lock().unwrap();
        let (queue, alive) = &mut *lock;
        if *alive {
            queue.push(val);
        }
        self.condvar.notify_one();
    }

    #[inline]
    pub fn deq(&self) -> Vec<T> {
        std::mem::take(
            &mut self
                .condvar
                .wait_while(self.queue.lock().unwrap(), |(q, a)| q.is_empty() && *a)
                .unwrap()
                .0,
        )
    }

    #[inline]
    pub fn shutdown(&self) {
        let mut lock = self.queue.lock().unwrap();
        let (queue, alive) = &mut *lock;
        *alive = false;
        queue.clear();
        self.condvar.notify_all();
    }
}
