use crate::{tensor, Storage};
use model_parameters::Llama2;
use tensor::{udim, Tensor};

pub struct LayerCache {
    /// Key cache, shape = `num_kv_head x max_seq_len x head_dim`.
    k: Tensor<Storage>,
    /// Value cache, shape = `num_kv_head x max_seq_len x head_dim`.
    v: Tensor<Storage>,
}

impl LayerCache {
    pub fn new_layers(model: &dyn Llama2) -> Vec<Self> {
        let dt = model.data_type();
        let nkvh = model.num_key_value_heads() as udim;
        let hd = (model.hidden_size() / model.num_attention_heads()) as udim;
        let max_seq_len = model.max_position_embeddings() as udim;
        let shape = &[nkvh, max_seq_len, hd];
        (0..model.num_hidden_layers())
            .map(|_| Self {
                k: tensor(dt, shape),
                v: tensor(dt, shape),
            })
            .collect()
    }

    #[inline]
    pub fn get(&mut self) -> (&mut Tensor<Storage>, &mut Tensor<Storage>) {
        (&mut self.k, &mut self.v)
    }
}
