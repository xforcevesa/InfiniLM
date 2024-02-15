use crate::{ConfigJson, DataType, LayerParamsOffset, Llama2, Memory};
use half::{bf16, f16};

impl Memory<Vec<u8>> {
    pub fn cast(src: &impl Llama2, new_dtype: DataType) -> Self {
        let from = src.data_type();
        let mut blob = Vec::with_capacity(src.size() * new_dtype.size() / from.size());
        let mut append = |src: &[u8]| {
            let start = blob.len();
            let additional = src.len() * new_dtype.size() / from.size();
            let end = start + additional;

            blob.reserve_exact(additional);
            unsafe { blob.set_len(end) };

            cast(from, src, new_dtype, &mut blob[start..end]);
            start
        };

        if let Some(n) = std::option_env!("RAYON_PARALLELISM") {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.parse().unwrap())
                .build_global()
                .unwrap();
        }

        let embed_tokens = append(src.embed_tokens());
        let layers = (0..src.num_hidden_layers())
            .map(|layer| LayerParamsOffset {
                input_layernorm: append(src.input_layernorm(layer)),
                self_attn_q_proj: append(src.self_attn_q_proj(layer)),
                self_attn_k_proj: append(src.self_attn_k_proj(layer)),
                self_attn_v_proj: append(src.self_attn_v_proj(layer)),
                self_attn_o_proj: append(src.self_attn_o_proj(layer)),
                post_attention_layernorm: append(src.post_attention_layernorm(layer)),
                mlp_gate: append(src.mlp_gate(layer)),
                mlp_down: append(src.mlp_down(layer)),
                mlp_up: append(src.mlp_up(layer)),
            })
            .collect();
        let model_norm = append(src.model_norm());
        let lm_head = append(src.lm_head());

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
            blob,
            embed_tokens,
            layers,
            model_norm,
            lm_head,
        }
    }
}

fn cast(src_dtype: DataType, src: &[u8], dst_dtype: DataType, dst: &mut [u8]) {
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

    match (src_dtype, dst_dtype) {
        (DataType::F16, DataType::F16)
        | (DataType::BF16, DataType::BF16)
        | (DataType::F32, DataType::F32) => dst.copy_from_slice(src),

        (DataType::F16, DataType::BF16) => {
            cast!(|x: f16| bf16::from_f32(x.to_f32()); src, f16 => dst, bf16);
        }
        (DataType::F16, DataType::F32) => {
            cast!(|x: f16| x.to_f32(); src, f16 => dst, f32);
        }
        (DataType::BF16, DataType::F16) => {
            cast!(|x: bf16| f16::from_f32(x.to_f32()); src, bf16 => dst, f16);
        }
        (DataType::BF16, DataType::F32) => {
            cast!(|x: bf16| x.to_f32(); src, bf16 => dst, f32);
        }
        (DataType::F32, DataType::F16) => {
            cast!(|x: f32| f16::from_f32(x); src, f32 => dst, f16);
        }
        (DataType::F32, DataType::BF16) => {
            cast!(|x: f32| bf16::from_f32(x); src, f32 => dst, bf16);
        }
    }
}
