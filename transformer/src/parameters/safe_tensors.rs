use super::{memory::Layer, storage::HostMem, ConfigJson, Memory, Storage};
use common::{
    safe_tensors::{
        Dtype, SafeTensors,
        SafeTensorsError::{self, Io, Json},
    },
    Blob,
};
use std::{fs::File, ops::DerefMut, path::Path, sync::Arc};
use tensor::{udim, DataType, Shape, Tensor};

impl Memory {
    pub fn load_safetensors(model_dir: impl AsRef<Path>) -> Result<Self, SafeTensorsError> {
        Self::load_safetensors_realloc(model_dir, Some(Blob::new))
    }

    pub fn load_safetensors_realloc<T: HostMem + DerefMut<Target = [u8]>>(
        model_dir: impl AsRef<Path>,
        mut realloc: Option<impl FnMut(usize) -> T>,
    ) -> Result<Self, SafeTensorsError> {
        let config = File::open(model_dir.as_ref().join("config.json")).map_err(Io)?;
        let config: ConfigJson = serde_json::from_reader(&config).map_err(Json)?;
        let model = SafeTensors::load_from_dir(model_dir)?.share();

        let tensor = |name: &str| {
            let shared = model
                .share_tensor(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let data_type = match shared.dtype() {
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
            assert_eq!(data_type, config.torch_dtype);
            Tensor::new(
                data_type,
                &shared.shape().iter().map(|&d| d as udim).collect::<Shape>(),
                Storage::SafeTensor(shared),
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
                            if model.contains(&qkv) {
                                tensor(&qkv)
                            } else if let Some(realloc) = realloc.as_mut() {
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
                                concat0(&[&q, &k, &v], realloc).reshape(&[d + dkv + dkv, d])
                            } else {
                                panic!("missing concat tensor: {qkv}");
                            }
                        },
                        self_attn_o_proj: tensor(&name("self_attn.o_proj")),
                        post_attention_layernorm: tensor(&name("post_attention_layernorm")),
                        mlp_gate_up: {
                            let gate_up = name("mlp.gate_up_proj");
                            if model.contains(&gate_up) {
                                tensor(&gate_up)
                            } else if let Some(realloc) = realloc.as_mut() {
                                concat0(
                                    &[
                                        &tensor(&name("mlp.gate_proj")),
                                        &tensor(&name("mlp.up_proj")),
                                    ],
                                    realloc,
                                )
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

fn concat0<T: HostMem + DerefMut<Target = [u8]>>(
    tensors: &[&Tensor<Storage>],
    realloc: impl FnOnce(usize) -> T,
) -> Tensor<Storage> {
    assert!(!tensors.is_empty());
    assert!(tensors
        .windows(2)
        .all(|t| t[0].data_type() == t[1].data_type()));

    let data_type = tensors[0].data_type();
    let mut shape = Shape::from_slice(tensors[0].shape());
    shape[0] = tensors.iter().map(|t| t.shape()[0]).sum();

    let mut ans = Tensor::alloc(data_type, &shape, realloc);
    let mut offset = 0;
    for t in tensors {
        let len = t.bytes_size();
        unsafe { t.reform_to_raw(&mut ans.physical_mut()[offset..][..len]) };
        offset += len;
    }
    ans.map_physical(|b| Storage::Others(Arc::new(b)))
}
