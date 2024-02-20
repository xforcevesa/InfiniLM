use crate::{memory::SafeTensorHeaderJson, ConfigJson, DataType, Llama2, Storage};
use safetensors::{tensor::TensorInfo, Dtype};
use std::{
    collections::HashMap,
    fs,
    io::{self, BufWriter, Write},
    path::Path,
};
use tensor::Tensor;

pub fn save(model: &dyn Llama2, dir: impl AsRef<Path>) -> io::Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let config = serde_json::to_string_pretty(&ConfigJson {
        bos_token_id: model.bos_token_id(),
        eos_token_id: model.eos_token_id(),
        hidden_size: model.hidden_size(),
        intermediate_size: model.intermediate_size(),
        max_position_embeddings: model.max_position_embeddings(),
        num_attention_heads: model.num_attention_heads(),
        num_hidden_layers: model.num_hidden_layers(),
        num_key_value_heads: model.num_key_value_heads(),
        vocab_size: model.vocab_size(),
        rms_norm_eps: model.rms_norm_eps(),
        rope_theta: model.rope_theta(),
        torch_dtype: model.data_type(),
    })?;
    fs::write(dir.join("config.json"), config)?;

    let mut offset = 0usize;
    let mut header = SafeTensorHeaderJson {
        tensors: HashMap::new(),
        meta: None,
    };

    let mut tensor_info = |tensor: Tensor<Storage>| TensorInfo {
        dtype: match tensor.data_type() {
            DataType::Bool => Dtype::BOOL,
            DataType::I8 => Dtype::I8,
            DataType::I16 => Dtype::I16,
            DataType::I32 => Dtype::I32,
            DataType::I64 => Dtype::I64,
            DataType::U8 => Dtype::U8,
            DataType::U16 => Dtype::U16,
            DataType::U32 => Dtype::U32,
            DataType::U64 => Dtype::U64,
            DataType::F16 => Dtype::F16,
            DataType::BF16 => Dtype::BF16,
            DataType::F32 => Dtype::F32,
            DataType::F64 => Dtype::F64,
        },
        shape: tensor.shape().iter().map(|&d| d as _).collect(),
        data_offsets: {
            let start = offset;
            offset += tensor.physical().as_slice().len();
            (start, offset)
        },
    };

    header.tensors.insert(
        "model.embed_tokens.weight".into(),
        tensor_info(model.embed_tokens()),
    );
    for layer in 0..model.num_hidden_layers() {
        header.tensors.insert(
            format!("model.layers.{layer}.input_layernorm.weight"),
            tensor_info(model.input_layernorm(layer)),
        );
        header.tensors.insert(
            format!("model.layers.{layer}.self_attn.qkv_proj.weight"),
            tensor_info(model.w_qkv(layer)),
        );
        header.tensors.insert(
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
            tensor_info(model.self_attn_o_proj(layer)),
        );
        header.tensors.insert(
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            tensor_info(model.post_attention_layernorm(layer)),
        );
        header.tensors.insert(
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            tensor_info(model.mlp_gate(layer)),
        );
        header.tensors.insert(
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            tensor_info(model.mlp_down(layer)),
        );
        header.tensors.insert(
            format!("model.layers.{layer}.mlp.up_proj.weight"),
            tensor_info(model.mlp_up(layer)),
        );
    }
    header
        .tensors
        .insert("model.norm.weight".into(), tensor_info(model.model_norm()));
    header
        .tensors
        .insert("lm_head.weight".into(), tensor_info(model.lm_head()));

    let mut file = fs::File::create(dir.join("model.safetensors"))?;
    let mut write = BufWriter::new(&mut file);
    {
        // write header
        let str = serde_json::to_string(&header)?;
        let len = str.len();
        const ALIGN: usize = std::mem::size_of::<usize>();
        let aligned = (len + ALIGN - 1) & !(ALIGN - 1);
        write.write_all(&(aligned as u64).to_le_bytes())?;
        write.write_all(str.as_bytes())?;
        for _ in len..aligned {
            write.write_all(&[32])?;
        }
    }
    write.write_all(model.embed_tokens().physical().as_slice())?;
    for layer in 0..model.num_hidden_layers() {
        write.write_all(model.input_layernorm(layer).physical().as_slice())?;
        write.write_all(model.w_qkv(layer).physical().as_slice())?;
        write.write_all(model.self_attn_o_proj(layer).physical().as_slice())?;
        write.write_all(model.post_attention_layernorm(layer).physical().as_slice())?;
        write.write_all(model.mlp_gate(layer).physical().as_slice())?;
        write.write_all(model.mlp_down(layer).physical().as_slice())?;
        write.write_all(model.mlp_up(layer).physical().as_slice())?;
    }
    write.write_all(model.model_norm().physical().as_slice())?;
    write.write_all(model.lm_head().physical().as_slice())?;
    Ok(())
}
