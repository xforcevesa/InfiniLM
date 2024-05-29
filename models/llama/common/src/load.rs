use crate::{
    json::{convert, ConfigJson},
    InferenceConfig, LayerStorage, Storage, Weight,
};
use common::{
    safe_tensors::SafeTensors,
    Blob,
    FileLoadError::{self, Io, Json},
};
use std::{fs::File, path::Path, pin::Pin, sync::Arc};
use tensor::{udim, DataType, Shape, Tensor};

impl Storage {
    pub fn load_safetensors(model_dir: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        let config = File::open(model_dir.as_ref().join("config.json")).map_err(Io)?;
        let config: ConfigJson = serde_json::from_reader(&config).map_err(Json)?;
        let model = SafeTensors::load_from_dir(model_dir)?.share();

        let dt = config.torch_dtype;
        let voc = config.vocab_size as udim;
        let d = config.hidden_size as udim;
        let nh = config.num_attention_heads as udim;
        let nkvh = config.num_key_value_heads as udim;
        let dh = d / nh;
        let dkv = dh * nkvh;
        let di = config.intermediate_size as udim;

        Ok(Self {
            config: InferenceConfig {
                dt,
                voc,
                nlayers: config.num_hidden_layers as _,
                nh,
                nkvh,
                d,
                dkv,
                di,
                max_seq_len: config.max_position_embeddings as _,
                bos_token: config.bos_token_id,
                eos_token: config.eos_token_id,
                epsilon: config.rms_norm_eps,
                theta: config.rope_theta,
            },

            embed_tokens: tensor(&model, "model.embed_tokens.weight", dt, [voc, d]),
            layers: (0..config.num_hidden_layers)
                .map(|l| {
                    let name = |name: &str| format!("model.layers.{l}.{name}.weight");
                    LayerStorage {
                        att_layernorm: tensor(&model, &name("input_layernorm"), dt, [d]),
                        att_qkv: {
                            let qkv = name("self_attn.qkv_proj");
                            if model.contains(&qkv) {
                                tensor(&model, &qkv, dt, [d + dkv + dkv, d])
                            } else {
                                let sq = &[nh, 2, dh / 2, d];
                                let skv = &[nkvh, 2, dh / 2, d];
                                let perm = &[0, 2, 1, 3];

                                let q = tensor(&model, &name("self_attn.q_proj"), dt, [d, d])
                                    .reshape(sq)
                                    .transpose(perm);
                                let k = tensor(&model, &name("self_attn.k_proj"), dt, [dkv, d])
                                    .reshape(skv)
                                    .transpose(perm);
                                let v = tensor(&model, &name("self_attn.v_proj"), dt, [dkv, d])
                                    .reshape(skv);
                                concat0(&[q, k, v]).reshape(&[d + dkv + dkv, d])
                            }
                        }
                        .transpose(&[1, 0]),
                        att_o: tensor(&model, &name("self_attn.o_proj"), dt, [d, d])
                            .transpose(&[1, 0]),
                        mlp_layernorm: tensor(&model, &name("post_attention_layernorm"), dt, [d]),
                        mlp_gate_up: {
                            let gate_up = name("mlp.gate_up_proj");
                            if model.contains(&gate_up) {
                                tensor(&model, &gate_up, dt, [di + di, d])
                            } else {
                                concat0(&[
                                    tensor(&model, &name("mlp.gate_proj"), dt, [di, d]),
                                    tensor(&model, &name("mlp.up_proj"), dt, [di, d]),
                                ])
                            }
                        }
                        .transpose(&[1, 0]),
                        mlp_down: tensor(&model, &name("mlp.down_proj"), dt, [d, di])
                            .transpose(&[1, 0]),
                    }
                })
                .collect(),
            lm_layernorm: tensor(&model, "model.norm.weight", dt, [d]),
            lm_head: tensor(&model, "lm_head.weight", dt, [voc, d]).transpose(&[1, 0]),
        })
    }
}

fn tensor<const N: usize>(
    model: &Pin<Arc<SafeTensors>>,
    name: &str,
    dt: DataType,
    shape: [udim; N],
) -> Tensor<Weight> {
    let shared = model
        .share_tensor(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    assert_eq!(convert!(Dtype: shared.dtype()), dt);
    assert_eq!(
        &*shared.shape().iter().map(|&d| d as udim).collect::<Shape>(),
        shape
    );
    Tensor::new(dt, &shape, Weight::SafeTensor(shared))
}

fn concat0(tensors: &[Tensor<Weight>]) -> Tensor<Weight> {
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
    ans.map_physical(|b| b.into())
}

#[test]
fn test_load() {
    if let Some(model_dir) = common::test_model::find() {
        println!("model_dir: {}", model_dir.display());

        let time = std::time::Instant::now();
        let _storage = Storage::load_safetensors(model_dir).unwrap();
        println!("load: {:?}", time.elapsed());
    };
}
