use crate::{
    json::{convert, ConfigJson},
    Storage, Weight,
};
use common::safe_tensors::{SafeTensorsHeader, SafeTensorsHeaderMetadata, TensorInfo};
use std::{
    collections::HashMap,
    fs,
    io::{self, BufWriter, Write},
    path::Path,
};
use tensor::Tensor;

impl Storage {
    pub fn save(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let config = serde_json::to_string_pretty(&ConfigJson {
            bos_token_id: self.config.bos_token,
            eos_token_id: self.config.eos_token,
            hidden_size: self.config.d as _,
            intermediate_size: self.config.di as _,
            max_position_embeddings: self.config.max_seq_len as _,
            num_attention_heads: self.config.nh as _,
            num_hidden_layers: self.config.nlayers as _,
            num_key_value_heads: self.config.nkvh as _,
            vocab_size: self.config.voc as _,
            rms_norm_eps: self.config.epsilon,
            rope_theta: self.config.theta,
            torch_dtype: self.config.dt,
        })?;
        fs::write(dir.join("config.json"), config)?;

        let mut offset = 0usize;
        let mut header = SafeTensorsHeader {
            tensors: HashMap::new(),
            metadata: SafeTensorsHeaderMetadata {
                format: "rs".into(),
            },
        };

        let mut t = |tensor: &Tensor<Weight>| TensorInfo {
            dtype: convert!(DataType: tensor.data_type()),
            shape: tensor.shape().iter().map(|&d| d as _).collect(),
            data_offsets: {
                let start = offset;
                offset += tensor.bytes_size();
                (start, offset)
            },
        };

        header
            .tensors
            .insert("model.embed_tokens.weight".into(), t(&self.embed_tokens));
        for (i, l) in self.layers.iter().enumerate() {
            #[rustfmt::skip]
            let iter = [
                ("input_layernorm"         , &l.att_layernorm),
                ("self_attn.qkv_proj"      , &l.att_qkv    .clone().transpose(&[1, 0])),
                ("self_attn.o_proj"        , &l.att_o      .clone().transpose(&[1, 0])),
                ("post_attention_layernorm", &l.mlp_layernorm),
                ("mlp.gate_up_proj"        , &l.mlp_gate_up.clone().transpose(&[1, 0])),
                ("mlp.down_proj"           , &l.mlp_down   .clone().transpose(&[1, 0])),
            ];
            header.tensors.extend(
                iter.map(|(name, tensor)| (format!("model.layers.{i}.{name}.weight"), t(tensor))),
            );
        }
        header.tensors.extend([
            ("model.norm.weight".into(), t(&self.lm_layernorm)),
            (
                "lm_head.weight".into(),
                t(&self.lm_head.clone().transpose(&[1, 0])),
            ),
        ]);

        let header = {
            let str = serde_json::to_string(&header)?;
            let len = str.len();
            const ALIGN: usize = std::mem::size_of::<usize>();
            let aligned = (len + ALIGN - 1) & !(ALIGN - 1);

            let mut buffer = Vec::with_capacity(aligned);
            let mut write = BufWriter::new(&mut buffer);
            write.write_all(&(aligned as u64).to_le_bytes())?;
            write.write_all(str.as_bytes())?;
            for _ in len..aligned {
                write.write_all(&[32])?;
            }
            drop(write);
            buffer
        };

        let mut file = fs::File::create(dir.join("model.safetensors"))?;
        file.write_all(&header)?;
        file.write_all(self.embed_tokens.physical())?;
        for l in self.layers.iter() {
            file.write_all(l.att_layernorm.physical())?;
            file.write_all(l.att_qkv.physical())?;
            file.write_all(l.att_o.physical())?;
            file.write_all(l.mlp_layernorm.physical())?;
            file.write_all(l.mlp_gate_up.physical())?;
            file.write_all(l.mlp_down.physical())?;
        }
        file.write_all(self.lm_layernorm.physical())?;
        file.write_all(self.lm_head.physical())?;
        Ok(())
    }
}
