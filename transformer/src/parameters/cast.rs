use super::{memory::Layer, ConfigJson, Llama2, Memory, Storage};
use half::{bf16, f16};
use tensor::{DataType, Tensor};

impl Memory {
    pub fn cast(src: &dyn Llama2, new_dtype: DataType) -> Self {
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
    if src.data_type() == new_dtype {
        return src;
    }

    assert!(src.is_contiguous());
    let src_data = src.as_slice();
    let mut data = vec![0u8; src.size() * new_dtype.size()];

    macro_rules! cast {
        ($f:expr; $src:expr, $src_ty:ty => $dst:expr, $dst_ty:ty) => {
            use rayon::iter::*;
            use std::{mem::size_of, slice::from_raw_parts, slice::from_raw_parts_mut};

            let len = $src.len() / size_of::<$src_ty>();
            debug_assert_eq!(len * size_of::<$dst_ty>(), $dst.len());
            let src = unsafe { from_raw_parts($src.as_ptr() as *const $src_ty, len) };
            let dst = unsafe { from_raw_parts_mut($dst.as_mut_ptr() as *mut $dst_ty, len) };

            #[allow(clippy::redundant_closure_call)]
            src.par_iter()
                .zip(dst)
                .for_each(|(src, dst)| *dst = $f(*src));
        };
    }

    match (src.data_type(), new_dtype) {
        (DataType::F16, DataType::BF16) => {
            cast!(|x: f16| bf16::from_f32(x.to_f32()); src_data, f16 => &mut data, bf16);
        }
        (DataType::F16, DataType::F32) => {
            cast!(|x: f16| x.to_f32(); src_data, f16 => &mut data, f32);
        }
        (DataType::BF16, DataType::F16) => {
            cast!(|x: bf16| f16::from_f32(x.to_f32()); src_data, bf16 => &mut data, f16);
        }
        (DataType::BF16, DataType::F32) => {
            cast!(|x: bf16| x.to_f32(); src_data, bf16 => &mut data, f32);
        }
        (DataType::F32, DataType::F16) => {
            cast!(|x: f32| f16::from_f32(x); src_data, f32 => &mut data, f16);
        }
        (DataType::F32, DataType::BF16) => {
            cast!(|x: f32| bf16::from_f32(x); src_data, f32 => &mut data, bf16);
        }
        _ => todo!(),
    }

    let pysical = Storage::from_blob(data);
    unsafe { Tensor::from_raw_parts(new_dtype, src.shape(), src.pattern(), pysical) }
}
