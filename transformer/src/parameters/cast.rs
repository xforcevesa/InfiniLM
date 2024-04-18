use super::{memory::Layer, ConfigJson, Llama2, Memory, Storage};
use common::Blob;
use half::{bf16, f16};
use std::sync::Arc;
use tensor::{DataType, Tensor, Ty};

impl Memory {
    pub fn cast(src: &(dyn Llama2), new_dtype: DataType) -> Self {
        Self {
            config: ConfigJson {
                torch_dtype: new_dtype,
                ..ConfigJson::from(src)
            },
            embed_tokens: cast(src.embed_tokens(), new_dtype),
            layers: (0..src.num_hidden_layers())
                .map(|l| Layer {
                    input_layernorm: cast(src.input_layernorm(l), new_dtype),
                    w_qkv: cast(src.w_qkv(l), new_dtype),
                    self_attn_o_proj: cast(src.self_attn_o_proj(l), new_dtype),
                    post_attention_layernorm: cast(src.post_attention_layernorm(l), new_dtype),
                    mlp_gate_up: cast(src.mlp_gate_up(l), new_dtype),
                    mlp_down: cast(src.mlp_down(l), new_dtype),
                })
                .collect(),
            model_norm: cast(src.model_norm(), new_dtype),
            lm_head: cast(src.lm_head(), new_dtype),
        }
    }
}

fn cast(src: Tensor<Storage>, new_dtype: DataType) -> Tensor<Storage> {
    match (src.data_type(), new_dtype) {
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
    src: Tensor<Storage>,
    cast: impl Fn(&T) -> U + Sync,
) -> Tensor<Storage> {
    use rayon::iter::*;
    use tensor::{reslice, reslice_mut};

    assert_eq!(src.data_type(), T::DATA_TYPE);
    if src.data_type() == U::DATA_TYPE {
        return src;
    }

    assert!(src.is_contiguous());
    let mut ans = Tensor::alloc(U::DATA_TYPE, src.shape(), Blob::new);

    reslice(src.physical())
        .par_iter()
        .zip(reslice_mut(ans.physical_mut()))
        .for_each(|(src, dst)| *dst = cast(src));

    unsafe { ans.map_physical(|b| Storage::Others(Arc::new(b))) }
}
