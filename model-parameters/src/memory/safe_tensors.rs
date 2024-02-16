use super::Memory;
use crate::{ConfigJson, DataType, LayerParamsOffset};
use memmap2::Mmap;
use safetensors::{tensor::TensorInfo, Dtype};
use std::{collections::HashMap, fs::File, path::Path};

#[derive(Debug)]
pub enum SafeTensorError {
    Io(std::io::Error),
    Serde(serde_json::Error),
}

impl Memory<Mmap> {
    pub fn load_safetensors(model_dir: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let dir = model_dir.as_ref();
        let config = File::open(dir.join("config.json")).map_err(SafeTensorError::Io)?;
        let model = File::open(dir.join("model.safetensors")).map_err(SafeTensorError::Io)?;

        let config: ConfigJson = serde_json::from_reader(config).map_err(SafeTensorError::Serde)?;
        let dtype = match config.torch_dtype {
            DataType::F16 => Dtype::F16,
            DataType::BF16 => Dtype::BF16,
            DataType::F32 => Dtype::F32,
        };

        let mmap = unsafe { Mmap::map(&model) }.map_err(SafeTensorError::Io)?;
        let len = unsafe { *mmap.as_ptr().cast::<u64>() } as usize;
        const BASE_OFFSET: usize = std::mem::size_of::<u64>();
        let header = &mmap[BASE_OFFSET..][..len];
        let header: SafeTensorHeaderJson =
            serde_json::from_slice(header).map_err(SafeTensorError::Serde)?;

        let d = config.hidden_size;
        let kv_dim = d * config.num_key_value_heads / config.num_attention_heads;
        let di = config.intermediate_size;

        let mut embed_tokens = 0;
        let mut layers = (0..config.num_hidden_layers)
            .map(|_| LayerParamsOffset {
                input_layernorm: 0,
                self_attn_q_proj: 0,
                self_attn_k_proj: 0,
                self_attn_v_proj: 0,
                self_attn_o_proj: 0,
                post_attention_layernorm: 0,
                mlp_gate: 0,
                mlp_down: 0,
                mlp_up: 0,
            })
            .collect::<Vec<_>>();
        let mut model_norm = 0;
        let mut lm_head = 0;

        let header_offset = BASE_OFFSET + len;
        for (name, tensor) in header.tensors {
            let path = name.split('.').collect::<Vec<_>>();
            let offset = header_offset + tensor.data_offsets.0;

            info!(target: "import safetensors", "detect {offset:#010x} -> \"{name}\"");
            match path.as_slice() {
                ["model", "embed_tokens", "weight"] => {
                    assert_eq!(&tensor.shape, &[config.vocab_size, d]);
                    assert_eq!(tensor.dtype, dtype);
                    embed_tokens = offset;
                }
                ["model", "layers", n, path @ .., "weight"] => {
                    let layer = n.parse::<usize>().unwrap();

                    match path {
                        ["input_layernorm"] => {
                            assert_eq!(&tensor.shape, &[d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].input_layernorm = offset;
                        }
                        ["self_attn", "q_proj"] => {
                            assert_eq!(&tensor.shape, &[d, d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].self_attn_q_proj = offset;
                        }
                        ["self_attn", "k_proj"] => {
                            assert_eq!(&tensor.shape, &[kv_dim, d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].self_attn_k_proj = offset;
                        }
                        ["self_attn", "v_proj"] => {
                            assert_eq!(&tensor.shape, &[kv_dim, d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].self_attn_v_proj = offset;
                        }
                        ["self_attn", "o_proj"] => {
                            assert_eq!(&tensor.shape, &[d, d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].self_attn_o_proj = offset;
                        }
                        ["post_attention_layernorm"] => {
                            assert_eq!(&tensor.shape, &[d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].post_attention_layernorm = offset;
                        }
                        ["mlp", "gate_proj"] => {
                            assert_eq!(&tensor.shape, &[di, d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].mlp_gate = offset;
                        }
                        ["mlp", "down_proj"] => {
                            assert_eq!(&tensor.shape, &[d, di]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].mlp_down = offset;
                        }
                        ["mlp", "up_proj"] => {
                            assert_eq!(&tensor.shape, &[di, d]);
                            assert_eq!(tensor.dtype, dtype);
                            layers[layer].mlp_up = offset;
                        }
                        [..] => {
                            warn!(target: "import safetensors", "Unknown tensor path: \"{name}\"")
                        }
                    };
                }
                ["model", "norm", "weight"] => {
                    assert_eq!(&tensor.shape, &[d]);
                    assert_eq!(tensor.dtype, dtype);
                    model_norm = offset;
                }
                ["lm_head", "weight"] => {
                    assert_eq!(&tensor.shape, &[config.vocab_size, d]);
                    assert_eq!(tensor.dtype, dtype);
                    lm_head = offset;
                }
                [..] => warn!(target: "import safetensors", "Unknown tensor path: \"{name}\""),
            }
        }

        Ok(Self {
            config,
            blob: mmap,
            embed_tokens,
            layers,
            model_norm,
            lm_head,
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
