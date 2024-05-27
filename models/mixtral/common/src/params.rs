use std::collections::HashMap;

use super::ConfigJson;
use common::{
    safe_tensors::{Dtype, SafeTensor, SafeTensors},
    Blob,
};
use tensor::{udim, DataType, Shape, Tensor};
pub struct MixtralParams {
    safe_tensors: SafeTensors,
    transformed_tensors: HashMap<String, Tensor<Blob>>,
}

impl MixtralParams {
    pub fn new(config: &ConfigJson, safe_tensors: SafeTensors) -> Self {
        let mut transformed_tensors: HashMap<String, Tensor<Blob>> = HashMap::new();
        let tensor_names = safe_tensors
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        let nh = config.num_attention_heads as udim;
        let nkvh = config.num_key_value_heads as udim;
        let d = config.hidden_size as udim;
        let dkv = d * nkvh / nh;
        let sq = &[nh, 2, d / nh / 2, d];
        let skv = &[nkvh, 2, dkv / nkvh / 2, d];
        let perm = &[0, 2, 1, 3];
        for name in tensor_names {
            if name.contains("q_proj") {
                let q = to_tensor(safe_tensors.get(&name).unwrap())
                    .reshape(sq)
                    .transpose(perm);
                let k = to_tensor(safe_tensors.get(&name.replace("q_proj", "k_proj")).unwrap())
                    .reshape(skv)
                    .transpose(perm);
                let v = to_tensor(safe_tensors.get(&name.replace("q_proj", "v_proj")).unwrap())
                    .reshape(skv);
                transformed_tensors.insert(
                    name.replace("q_proj", "qkv_proj"),
                    concat0(&[q, k, v]).reshape(&[d + dkv + dkv, d]),
                );
            } else if name.contains("w1") {
                let w1 = to_tensor(safe_tensors.get(&name).unwrap());
                let w3 = to_tensor(safe_tensors.get(&name.replace("w1", "w3")).unwrap());
                transformed_tensors.insert(name.replace("w1", "gate_up_proj"), concat0(&[w1, w3]));
            }
        }
        Self {
            safe_tensors,
            transformed_tensors,
        }
    }
}

impl MixtralParams {
    pub fn embed_tokens(&self) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, "model.embed_tokens.weight")
    }

    pub fn input_layernorm(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "input_layernorm"))
    }

    pub fn w_qkv(&self, layer: udim) -> Tensor<&[u8]> {
        let name = layer_name(layer, "self_attn.qkv_proj");
        if let Some(t) = self.safe_tensors.get(&name) {
            to_tensor(t)
        } else {
            self.transformed_tensors
                .get(&name)
                .unwrap()
                .as_ref()
                .map_physical(|u| &**u)
        }
    }

    pub fn w_o(&self, layer: udim) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, layer_name(layer, "self_attn.o_proj"))
    }

    pub fn post_attention_layernorm(&self, layer: udim) -> Tensor<&[u8]> {
        convert(
            &self.safe_tensors,
            layer_name(layer, "post_attention_layernorm"),
        )
    }

    pub fn moe_gate(&self, layer: udim) -> Tensor<&[u8]> {
        convert(
            &self.safe_tensors,
            layer_name(layer, "block_sparse_moe.gate"),
        )
    }

    pub fn mlp_gate_up(&self, layer: udim, expert: udim) -> Tensor<&[u8]> {
        let name = layer_name(
            layer,
            &format!("block_sparse_moe.experts.{}.gate_up_proj", expert),
        );
        if let Some(t) = self.safe_tensors.get(&name) {
            to_tensor(t)
        } else {
            self.transformed_tensors
                .get(&name)
                .unwrap()
                .as_ref()
                .map_physical(|u| &**u)
        }
    }

    pub fn mlp_down(&self, layer: udim, expert: udim) -> Tensor<&[u8]> {
        convert(
            &self.safe_tensors,
            layer_name(layer, &format!("block_sparse_moe.experts.{}.w2", expert)),
        )
    }

    pub fn model_norm(&self) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, "model.norm.weight")
    }

    pub fn lm_head(&self) -> Tensor<&[u8]> {
        convert(&self.safe_tensors, "lm_head.weight")
    }
}

fn layer_name(layer: udim, name: &str) -> String {
    format!("model.layers.{layer}.{name}.weight")
}

fn to_tensor(tensor: SafeTensor) -> Tensor<&[u8]> {
    let data_type = type_convert(tensor.dtype);
    let shape = tensor.shape.iter().map(|&x| x as udim).collect::<Vec<_>>();
    Tensor::new(data_type, &shape, tensor.data)
}

fn convert(tensors: &SafeTensors, name: impl AsRef<str>) -> Tensor<&[u8]> {
    let tensor = tensors
        .get(name.as_ref())
        .unwrap_or_else(|| panic!("Tensor {} not found", name.as_ref()));
    let data_type = type_convert(tensor.dtype);
    let shape = tensor.shape.iter().map(|&x| x as udim).collect::<Vec<_>>();
    Tensor::new(data_type, &shape, tensor.data)
}

pub fn type_convert(dtype: Dtype) -> DataType {
    match dtype {
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
    }
}

fn concat0(tensors: &[Tensor<&[u8]>]) -> Tensor<Blob> {
    assert!(tensors
        .windows(2)
        .all(|t| t[0].data_type() == t[1].data_type()));
    assert!(!tensors.is_empty());

    let data_type = tensors[0].data_type();
    let mut shape = Shape::from_slice(tensors[0].shape());
    shape[0] = tensors.iter().map(|t| t.shape()[0]).sum();

    let mut ans = Tensor::alloc(data_type, &shape, Blob::new);
    let mut offset = 0;
    for t in tensors {
        let len = t.bytes_size();
        unsafe { t.reform_to_raw(&mut ans.physical_mut()[offset..][..len]) };
        offset += len;
    }
    ans.map_physical(|b| b)
}
