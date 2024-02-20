use super::{Layer, Memory};
use crate::{ConfigJson, DataType, Storage};
use memmap2::Mmap;
use safetensors::{tensor::TensorInfo, Dtype};
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};
use tensor::Tensor;

#[derive(Debug)]
pub enum SafeTensorError {
    Io(std::io::Error),
    Serde(serde_json::Error),
}

impl Memory {
    pub fn load_safetensors(model_dir: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let dir = model_dir.as_ref();
        let config = File::open(dir.join("config.json")).map_err(SafeTensorError::Io)?;
        let model = File::open(dir.join("model.safetensors")).map_err(SafeTensorError::Io)?;

        let config: ConfigJson = serde_json::from_reader(config).map_err(SafeTensorError::Serde)?;

        let mmap = unsafe { Mmap::map(&model) }.map_err(SafeTensorError::Io)?;
        let len = unsafe { *mmap.as_ptr().cast::<u64>() } as usize;
        const BASE_OFFSET: usize = std::mem::size_of::<u64>();
        let header = &mmap[BASE_OFFSET..][..len];
        let header: SafeTensorHeaderJson =
            serde_json::from_slice(header).map_err(SafeTensorError::Serde)?;

        let mmap = Arc::new(mmap);
        let tensor = |name: &str| {
            let info = &header.tensors[name];
            let (start, end) = info.data_offsets;
            Tensor::new(
                match info.dtype {
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
                },
                info.shape.iter().map(|&d| d as _).collect(),
                Storage::new(mmap.clone(), start, end - start),
            )
        };

        Ok(Self {
            embed_tokens: tensor("model.embed_tokens.weight"),
            layers: (0..config.num_hidden_layers)
                .map(|l| {
                    let name = |name: &str| format!("model.layers.{l}.{name}.weight");
                    Layer {
                        input_layernorm: tensor(&name("input_layernorm")),
                        self_attn_q_proj: tensor(&name("self_attn.q_proj")),
                        self_attn_k_proj: tensor(&name("self_attn.k_proj")),
                        self_attn_v_proj: tensor(&name("self_attn.v_proj")),
                        self_attn_o_proj: tensor(&name("self_attn.o_proj")),
                        post_attention_layernorm: tensor(&name("post_attention_layernorm")),
                        mlp_gate: tensor(&name("mlp.gate_proj")),
                        mlp_down: tensor(&name("mlp.down_proj")),
                        mlp_up: tensor(&name("mlp.up_proj")),
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
