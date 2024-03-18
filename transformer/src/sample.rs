use std::{cmp::Ordering, collections::BinaryHeap};

use common::utok;
use half::f16;

#[derive(Clone, PartialEq, Debug)]
pub enum SampleArgs {
    Top,
    Random {
        temperature: f32,
        top_k: usize,
        top_p: f32,
    },
}

pub fn argmax<T: PartialOrd>(logits: &[T]) -> utok {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as _
}

pub fn random(logits: &[f16], temperature: f32, top_k: usize, top_p: f32) -> utok {
    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: utok,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            match self.val.partial_cmp(&other.val).unwrap() {
                Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    // top-k & max
    let (logits, max) = {
        let mut buf = BinaryHeap::with_capacity(top_k + 1);
        let mut max = f32::NEG_INFINITY;
        for (i, p) in logits.iter().enumerate() {
            let val = p.to_f32();
            max = max.max(val);
            buf.push(Probability { val, tok: i as _ });
            if buf.len() > top_k {
                buf.pop();
            }
        }
        (buf.into_vec(), max)
    };
    // temperature & sum
    let (logits, sum) = {
        let mut logits = logits;
        let mut sum = 0.;
        for pi in logits.iter_mut() {
            pi.val = ((pi.val - max) / temperature).exp();
            sum += pi.val;
        }
        (logits, sum)
    };
    // top p
    let logits = if (0. ..1.).contains(&top_p) {
        let i = logits
            .iter()
            .scan(top_p * sum, |top_p, pi| {
                if *top_p > 0. {
                    *top_p -= pi.val;
                    Some(())
                } else {
                    None
                }
            })
            .count();
        &logits[..i]
    } else {
        &logits[..]
    };
    // random
    let mut rand = rand::random::<f32>() * sum;
    logits
        .iter()
        .find(|pi| {
            rand -= pi.val;
            rand <= 0.
        })
        .unwrap_or(logits.last().unwrap())
        .tok
}
