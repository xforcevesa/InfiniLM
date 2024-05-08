use common::{utok, BetweenF32};
use std::{cmp::Ordering, mem::replace};

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
                match self.val.total_cmp(&other.val) {
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

        // sort
        let mut logits = logits
            .iter()
            .enumerate()
            .map(Probability::from)
            .collect::<Vec<_>>();
        logits.sort_unstable();
        let max = replace(&mut logits[0].val, 1.);
        // softmax & sum
        for i in 1..logits.len() {
            logits[i].val = logits[i - 1].val + ((logits[i].val - max) / self.temperature).exp();
        }
        // topk & topp & random
        let pk = logits[self.top_k.min(logits.len()) - 1].val;
        let pp = logits[logits.len() - 1].val * self.top_p;
        let plimit = rand::random::<f32>() * f32::min(pk, pp);
        // sample
        logits.iter().find(|p| p.val >= plimit).unwrap().tok
    }
}
