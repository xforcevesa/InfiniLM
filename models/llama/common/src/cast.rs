use crate::{InferenceConfig, LayerStorage, Storage, Weight};
use common::{bf16, f16, Blob};
use tensor::{DataType, Tensor, Ty};

impl Storage {
    pub fn cast(self, dt: DataType) -> Self {
        if self.config.dt == dt {
            return self;
        }
        Self {
            config: InferenceConfig { dt, ..self.config },
            embed_tokens: cast(self.embed_tokens, dt),
            layers: self
                .layers
                .into_iter()
                .map(|l| LayerStorage {
                    att_layernorm: cast(l.att_layernorm, dt),
                    att_qkv: cast(l.att_qkv, dt),
                    att_o: cast(l.att_o, dt),
                    mlp_layernorm: cast(l.mlp_layernorm, dt),
                    mlp_gate_up: cast(l.mlp_gate_up, dt),
                    mlp_down: cast(l.mlp_down, dt),
                })
                .collect(),
            lm_layernorm: cast(self.lm_layernorm, dt),
            lm_head: cast(self.lm_head, dt),
        }
    }
}

fn cast(src: Tensor<Weight>, dt: DataType) -> Tensor<Weight> {
    match (src.data_type(), dt) {
        (DataType::F16, DataType::BF16) => typed(src, |x: &f16| bf16::from_f32(x.to_f32())),
        (DataType::F16, DataType::F32) => typed(src, |x: &f16| x.to_f32()),
        (DataType::BF16, DataType::F16) => typed(src, |x: &bf16| f16::from_f32(x.to_f32())),
        (DataType::BF16, DataType::F32) => typed(src, |x: &bf16| x.to_f32()),
        (DataType::F32, DataType::F16) => typed(src, |x: &f32| f16::from_f32(*x)),
        (DataType::F32, DataType::BF16) => typed(src, |x: &f32| bf16::from_f32(*x)),
        _ => todo!(),
    }
}

fn typed<T: Ty + Sync, U: Ty + Send>(
    src: Tensor<Weight>,
    cast: impl Fn(&T) -> U + Sync,
) -> Tensor<Weight> {
    use rayon::iter::*;
    use tensor::{reslice, reslice_mut};

    assert_eq!(src.data_type(), T::DATA_TYPE);
    let mut ans = Tensor::alloc(U::DATA_TYPE, src.shape(), Blob::new);

    reslice(src.physical())
        .par_iter()
        .zip(reslice_mut(ans.physical_mut()))
        .for_each(|(src, dst)| *dst = cast(src));

    ans.map_physical(|b| b.into())
}
