mod config;

use crate::{DataType, LLama2};
use config::{ConfigJson, SafeTensorHeaderJson};
use log::warn;
use memmap2::Mmap;
use safetensors::Dtype;
use std::{fs::File, path::Path};

pub struct SafeTensors {
    config: ConfigJson,
    mmap: Mmap,
    embed_tokens: usize,
    layers: Vec<LayerParamsOffset>,
    model_norm: usize,
    lm_head: usize,
}

struct LayerParamsOffset {
    input_layernorm: usize,
    self_attn_q_proj: usize,
    self_attn_k_proj: usize,
    self_attn_v_proj: usize,
    self_attn_o_proj: usize,
    post_attention_layernorm: usize,
    mlp_gate: usize,
    mlp_down: usize,
    mlp_up: usize,
}

macro_rules! slice {
    ($mmap:expr; $offset:expr, $len:expr) => {
        $mmap[$offset..][..$len]
    };
}

impl LLama2 for SafeTensors {
    #[inline]
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[inline]
    fn intermediate_size(&self) -> usize {
        self.config.intermediate_size
    }

    #[inline]
    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    #[inline]
    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    #[inline]
    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    #[inline]
    fn num_key_value_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    #[inline]
    fn data_type(&self) -> DataType {
        match self.config.torch_dtype.as_str() {
            "float16" => DataType::F16,
            "bfloat16" => DataType::BF16,
            "float32" => DataType::F32,
            t => panic!("Unsupported dtype: \"{t}\""),
        }
    }

    #[inline]
    fn embed_tokens(&self) -> &[u8] {
        let d = self.config.hidden_size;
        let dv = self.config.vocab_size;
        let dt = self.data_type().size();
        &slice!(self.mmap; self.embed_tokens, dv * d * dt)
    }

    #[inline]
    fn input_layernorm(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].input_layernorm, d * dt)
    }

    #[inline]
    fn self_attn_q_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].self_attn_q_proj, d * d * dt)
    }

    #[inline]
    fn self_attn_k_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dkv = d * self.config.num_key_value_heads / self.config.num_attention_heads;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].self_attn_k_proj, dkv * d * dt)
    }

    #[inline]
    fn self_attn_v_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dkv = d * self.config.num_key_value_heads / self.config.num_attention_heads;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].self_attn_v_proj, dkv * d * dt)
    }

    #[inline]
    fn self_attn_o_proj(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].self_attn_o_proj, d * d * dt)
    }

    #[inline]
    fn post_attention_layernorm(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].post_attention_layernorm, d * dt)
    }

    #[inline]
    fn mlp_gate(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let di = self.config.intermediate_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].mlp_gate, di * d * dt)
    }

    #[inline]
    fn mlp_down(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let di = self.config.intermediate_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].mlp_down, d * di * dt)
    }

    #[inline]
    fn mlp_up(&self, layer: usize) -> &[u8] {
        let d = self.config.hidden_size;
        let di = self.config.intermediate_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.layers[layer].mlp_up, di * d * dt)
    }

    #[inline]
    fn model_norm(&self) -> &[u8] {
        let d = self.config.hidden_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.model_norm, d * dt)
    }

    #[inline]
    fn lm_head(&self) -> &[u8] {
        let d = self.config.hidden_size;
        let dv: usize = self.config.vocab_size;
        let dt: usize = self.data_type().size();
        &slice!(self.mmap; self.lm_head, dv * d * dt)
    }
}

#[derive(Debug)]
pub enum SafeTensorError {
    Io(std::io::Error),
    Serde(serde_json::Error),
}

impl SafeTensors {
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let dir = model_dir.as_ref();
        let config = File::open(dir.join("config.json")).map_err(SafeTensorError::Io)?;
        let model = File::open(dir.join("model.safetensors")).map_err(SafeTensorError::Io)?;

        let config: ConfigJson = serde_json::from_reader(config).map_err(SafeTensorError::Serde)?;
        let dtype = match config.torch_dtype.as_str() {
            "float16" => Dtype::F16,
            "bfloat16" => Dtype::BF16,
            "float32" => Dtype::F32,
            _ => panic!("Unsupported dtype: {}", config.torch_dtype),
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

        for (name, tensor) in header.tensors {
            let path = name.split('.').collect::<Vec<_>>();
            let offset = BASE_OFFSET + len + tensor.data_offsets.0;

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
            mmap,
            embed_tokens,
            layers,
            model_norm,
            lm_head,
        })
    }
}

#[test]
fn test_load() {
    use std::time::Instant;
    let t0 = Instant::now();
    let safetensors = SafeTensors::new("../../TinyLlama-1.1B-Chat-v1.0");
    let t1 = Instant::now();
    println!("{:?}", t1 - t0);
    match safetensors {
        Ok(_) => {}
        Err(SafeTensorError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => panic!("{:?}", e),
    }
}
