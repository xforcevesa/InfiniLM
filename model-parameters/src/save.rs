use crate::{memory::SafeTensorHeaderJson, ConfigJson, DataType, Llama2};
use safetensors::{tensor::TensorInfo, Dtype};
use std::{
    collections::HashMap,
    fs,
    io::{self, BufWriter, Write},
    path::Path,
};

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

    let dtype = match model.data_type() {
        DataType::F16 => Dtype::F16,
        DataType::BF16 => Dtype::BF16,
        DataType::F32 => Dtype::F32,
        _ => todo!(),
    };
    let d = model.hidden_size();
    let dkv = d * model.num_key_value_heads() / model.num_attention_heads();
    let di = model.intermediate_size();
    let dv = model.vocab_size();

    struct Offset(usize);
    impl Offset {
        #[inline]
        fn update(&mut self, len: usize) -> (usize, usize) {
            let start = self.0;
            self.0 += len;
            (start, self.0)
        }
    }

    let mut offset = Offset(0);
    let mut header = SafeTensorHeaderJson {
        tensors: HashMap::new(),
        meta: None,
    };
    header.tensors.insert(
        "model.embed_tokens.weight".into(),
        TensorInfo {
            dtype,
            shape: vec![dv, d],
            data_offsets: offset.update(model.embed_tokens().physical().as_slice().len()),
        },
    );
    for layer in 0..model.num_hidden_layers() {
        header.tensors.insert(
            format!("model.layers.{layer}.input_layernorm.weight",),
            TensorInfo {
                dtype,
                shape: vec![d],
                data_offsets: offset
                    .update(model.input_layernorm(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![d, d],
                data_offsets: offset
                    .update(model.self_attn_q_proj(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.self_attn.k_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![dkv, d],
                data_offsets: offset
                    .update(model.self_attn_k_proj(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.self_attn.v_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![dkv, d],
                data_offsets: offset
                    .update(model.self_attn_v_proj(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![d, d],
                data_offsets: offset
                    .update(model.self_attn_o_proj(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            TensorInfo {
                dtype,
                shape: vec![d],
                data_offsets: offset.update(
                    model
                        .post_attention_layernorm(layer)
                        .physical()
                        .as_slice()
                        .len(),
                ),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![di, d],
                data_offsets: offset.update(model.mlp_gate(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![d, di],
                data_offsets: offset.update(model.mlp_down(layer).physical().as_slice().len()),
            },
        );
        header.tensors.insert(
            format!("model.layers.{layer}.mlp.up_proj.weight"),
            TensorInfo {
                dtype,
                shape: vec![di, d],
                data_offsets: offset.update(model.mlp_up(layer).physical().as_slice().len()),
            },
        );
    }
    header.tensors.insert(
        "model.norm.weight".into(),
        TensorInfo {
            dtype,
            shape: vec![d],
            data_offsets: offset.update(model.model_norm().physical().as_slice().len()),
        },
    );
    header.tensors.insert(
        "lm_head.weight".into(),
        TensorInfo {
            dtype,
            shape: vec![dv, d],
            data_offsets: offset.update(model.lm_head().physical().as_slice().len()),
        },
    );

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
        write.write_all(model.self_attn_q_proj(layer).physical().as_slice())?;
        write.write_all(model.self_attn_k_proj(layer).physical().as_slice())?;
        write.write_all(model.self_attn_v_proj(layer).physical().as_slice())?;
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
