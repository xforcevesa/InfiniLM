use crate::{kernel, Storage};
use common::utok;
use gemm::f16;
use tensor::{reslice, DataType, Tensor};
use transformer::SampleArgs;

pub struct Sample;

impl transformer::Sample<Storage> for Sample {
    fn sample<Id>(
        &self,
        args: &SampleArgs,
        requests: Vec<Id>,
        logits: Tensor<Storage>,
    ) -> Vec<(Id, utok)> {
        let &[_, voc] = logits.shape() else { panic!() };
        let dt = logits.data_type();

        macro_rules! sample {
                ($ty:ty) => {{
                    let logits: &[$ty] = reslice(logits.as_slice());
                    requests
                        .into_iter()
                        .enumerate()
                        .map(|(i, id)| (id, args.random(&kernel::slice!(logits; voc; [i]))))
                        .collect()
                }};
            }

        match dt {
            DataType::F16 => sample!(f16),
            DataType::F32 => sample!(f32),
            _ => unreachable!(),
        }
    }
}
