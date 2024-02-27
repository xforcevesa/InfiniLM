use super::{Layer, Memory};
use crate::{memory::concat0, ConfigJson, DataType, Storage};
use memmap2::Mmap;
use safetensors::{tensor::TensorInfo, Dtype};
use std::{collections::HashMap, fs::File, io::Read, ops::Deref, path::Path, sync::Arc};
use tensor::{udim, Tensor};

#[derive(Debug)]
pub enum SafeTensorError {
    Io(std::io::Error),
    Serde(serde_json::Error),
}

impl Memory {
    pub fn load_safetensors_from_dir(model_dir: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let model_dir = model_dir.as_ref();
        let config = File::open(model_dir.join("config.json")).map_err(SafeTensorError::Io)?;
        let model = File::open(model_dir.join("model.safetensors")).map_err(SafeTensorError::Io)?;
        let model = unsafe { Mmap::map(&model) }.map_err(SafeTensorError::Io)?;
        Ok(Self::load_safetensors(config, model, true).map_err(SafeTensorError::Serde)?)
    }

    pub fn load_safetensors(
        config: impl Read,
        model: impl Deref<Target = [u8]> + 'static,
        allow_realloc: bool,
    ) -> Result<Self, serde_json::Error> {
        let config: ConfigJson = serde_json::from_reader(config)?;

        let len = unsafe { *model.as_ptr().cast::<u64>() } as usize;
        const BASE_OFFSET: usize = std::mem::size_of::<u64>();
        let header = &model[BASE_OFFSET..][..len];
        let header: SafeTensorHeaderJson = serde_json::from_slice(header)?;

        let mmap = Arc::new(model);
        let offset = BASE_OFFSET + len;
        let tensor = |name: &str| {
            let info = &header.tensors[name];
            let (start, end) = info.data_offsets;
            let data_type = match info.dtype {
                Dtype::BOOL => DataType::Bool,
                Dtype::I8 => DataType::I8,
                Dtype::I16 => DataType::I16,
                Dtype::I32 => DataType::I32,
                Dtype::I64 => DataType::I64,
                Dtype::U8 => DataType::U8,
                Dtype::U16 => DataType::U16,
                Dtype::U32 => DataType::U32,
                Dtype::U64 => DataType::U64,
                Dtype::F16 => DataType::F16,
                Dtype::BF16 => DataType::BF16,
                Dtype::F32 => DataType::F32,
                Dtype::F64 => DataType::F64,
                _ => unreachable!(),
            };
            debug_assert_eq!(data_type, config.torch_dtype);
            Tensor::new(
                data_type,
                &info.shape.iter().map(|&d| d as udim).collect::<Vec<_>>(),
                Storage::new(mmap.clone(), offset + start, end - start),
            )
        };

        Ok(Self {
            embed_tokens: tensor("model.embed_tokens.weight"),
            layers: (0..config.num_hidden_layers)
                .map(|l| {
                    let name = |name: &str| format!("model.layers.{l}.{name}.weight");
                    Layer {
                        input_layernorm: tensor(&name("input_layernorm")),
                        w_qkv: {
                            let qkv = name("self_attn.qkv_proj");
                            if header.tensors.contains_key(&qkv) {
                                tensor(&qkv)
                            } else if allow_realloc {
                                let d = config.hidden_size as udim;
                                let nkvh = config.num_key_value_heads as udim;
                                let nh = config.num_attention_heads as udim;
                                let dkv = d * nkvh / nh;
                                let sq = &[nh, 2, d / nh / 2, d];
                                let skv = &[nkvh, 2, dkv / nkvh / 2, d];
                                let perm = &[0, 2, 1, 3];

                                let q = tensor(&name("self_attn.q_proj"))
                                    .reshape(sq)
                                    .transpose(perm);
                                let k = tensor(&name("self_attn.k_proj"))
                                    .reshape(skv)
                                    .transpose(perm);
                                let v = tensor(&name("self_attn.v_proj")).reshape(skv);
                                concat0(&[&q, &k, &v]).reshape(&[d + dkv + dkv, d])
                            } else {
                                panic!("missing concat tensor: {qkv}");
                            }
                        },
                        self_attn_o_proj: tensor(&name("self_attn.o_proj")),
                        post_attention_layernorm: tensor(&name("post_attention_layernorm")),
                        mlp_gate_up: {
                            let gate_up = name("mlp.gate_up_proj");
                            if header.tensors.contains_key(&gate_up) {
                                tensor(&gate_up)
                            } else if allow_realloc {
                                concat0(&[
                                    &tensor(&name("mlp.gate_proj")),
                                    &tensor(&name("mlp.up_proj")),
                                ])
                            } else {
                                panic!("missing concat tensor: {gate_up}");
                            }
                        },
                        mlp_down: tensor(&name("mlp.down_proj")),
                    }
                })
                .collect(),
            model_norm: tensor("model.norm.weight"),
            lm_head: tensor("lm_head.weight"),
            config,
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct SafeTensorHeaderJson {
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(rename = "__metadata__")]
    pub meta: Option<HashMap<String, serde_json::Value>>,
}
