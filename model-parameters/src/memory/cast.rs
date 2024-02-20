use super::Layer;
use crate::{ConfigJson, DataType, Llama2, Memory, Storage};
use half::{bf16, f16};
use tensor::Tensor;

impl Memory {
    pub fn cast(src: &dyn Llama2, new_dtype: DataType) -> Self {
        Self {
            config: ConfigJson {
                bos_token_id: src.bos_token_id(),
                eos_token_id: src.eos_token_id(),
                hidden_size: src.hidden_size(),
                intermediate_size: src.intermediate_size(),
                max_position_embeddings: src.max_position_embeddings(),
                num_attention_heads: src.num_attention_heads(),
                num_hidden_layers: src.num_hidden_layers(),
                num_key_value_heads: src.num_key_value_heads(),
                vocab_size: src.vocab_size(),
                rms_norm_eps: src.rms_norm_eps(),
                rope_theta: src.rope_theta(),
                torch_dtype: new_dtype,
            },
            embed_tokens: cast(src.embed_tokens(), new_dtype),
            layers: (0..src.num_hidden_layers())
                .map(|l| Layer {
                    input_layernorm: cast(src.input_layernorm(l), new_dtype),
                    self_attn_q_proj: cast(src.self_attn_q_proj(l), new_dtype),
                    self_attn_k_proj: cast(src.self_attn_k_proj(l), new_dtype),
                    self_attn_v_proj: cast(src.self_attn_v_proj(l), new_dtype),
                    self_attn_o_proj: cast(src.self_attn_o_proj(l), new_dtype),
                    post_attention_layernorm: cast(src.post_attention_layernorm(l), new_dtype),
                    mlp_gate: cast(src.mlp_gate(l), new_dtype),
                    mlp_down: cast(src.mlp_down(l), new_dtype),
                    mlp_up: cast(src.mlp_up(l), new_dtype),
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

    let src_data = src.physical().as_slice();
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
    unsafe { src.cast(new_dtype, pysical) }
}
