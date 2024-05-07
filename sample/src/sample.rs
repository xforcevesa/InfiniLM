use common::{utok, BetweenF32};
use std::{cmp::Ordering, collections::BinaryHeap, fmt::Debug};

impl crate::SampleArgs {
    #[inline]
    pub fn is_argmax(&self) -> bool {
        self.temperature <= 0. || self.top_k < 2 || self.top_p <= 0.
    }

    pub fn random<T>(&self, logits: &[T]) -> utok
    where
        T: BetweenF32 + PartialOrd,
    {
        if self.is_argmax() {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as _;
        }

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
        impl<T: BetweenF32> From<(usize, &T)> for Probability {
            #[inline]
            fn from((i, p): (usize, &T)) -> Self {
                Self {
                    val: p.get(),
                    tok: i as _,
                }
            }
        }

        // top-k & max
        let logits = if self.top_k < logits.len() {
            let mut buf = BinaryHeap::with_capacity(self.top_k + 1);
            for it in logits.iter().enumerate() {
                buf.push(Probability::from(it));
                if buf.len() > self.top_k {
                    buf.pop();
                }
            }
            buf.into_vec()
        } else {
            let mut buf = logits
                .iter()
                .enumerate()
                .map(Probability::from)
                .collect::<Vec<_>>();
            buf.sort_unstable();
            buf
        };
        let max = logits[0].val;
        // temperature & sum
        let (logits, sum) = {
            let mut logits = logits;
            let mut sum = 0.;
            for pi in logits.iter_mut() {
                pi.val = ((pi.val - max) / self.temperature).exp();
                sum += pi.val;
            }
            (logits, sum)
        };
        // top p
        let logits = if self.top_p < 1. {
            let i = logits
                .iter()
                .scan(self.top_p * sum, |top_p, pi| {
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
}
