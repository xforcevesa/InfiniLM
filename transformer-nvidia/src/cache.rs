use crate::tensor;
use cuda::{LocalDevBlob, Stream};
use model_parameters::Llama2;
use tensor::{udim, Tensor};

pub struct LayerCache<'a> {
    /// Key cache, shape = `num_kv_head x max_seq_len x head_dim`.
    k: Tensor<LocalDevBlob<'a>>,
    /// Value cache, shape = `num_kv_head x max_seq_len x head_dim`.
    v: Tensor<LocalDevBlob<'a>>,
}

impl<'a> LayerCache<'a> {
    pub fn new_layers(model: &dyn Llama2, stream: &'a Stream) -> Vec<Self> {
        let dt = model.data_type();
        let nkvh = model.num_key_value_heads() as udim;
        let hd = (model.hidden_size() / model.num_attention_heads()) as udim;
        let max_seq_len = model.max_position_embeddings() as udim;
        let shape = &[nkvh, max_seq_len, hd];
        (0..model.num_hidden_layers())
            .map(|_| Self {
                k: tensor(dt, shape, stream),
                v: tensor(dt, shape, stream),
            })
            .collect()
    }

    #[inline]
    pub fn get(&self) -> (&'a Tensor<LocalDevBlob>, &'a Tensor<LocalDevBlob>) {
        (&self.k, &self.v)
    }
}
