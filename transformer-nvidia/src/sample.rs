use crate::Storage;
use common::utok;
use half::f16;
use tensor::Tensor;
use transformer::SampleArgs;

pub struct Sample;

impl<'ctx> transformer::Sample<Storage<'ctx>> for Sample {
    fn sample<Id>(
        &self,
        args: &SampleArgs,
        requests: Vec<Id>,
        logits: Tensor<Storage>,
    ) -> Vec<(Id, utok)> {
        assert_eq!(logits.data_type(), tensor::DataType::F16);
        let &[_, voc] = logits.shape() else { panic!() };

        let mut host = vec![f16::ZERO; logits.size()];
        logits.physical().copy_out(&mut host);

        requests
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, args.random(&host[i * voc as usize..][..voc as usize])))
            .collect()
    }
}
