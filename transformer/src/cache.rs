use model_parameters::Llama2;
use tensor::{udim, DataType, Tensor};

/// KV cache for one layer.
pub struct LayerCache<Storage> {
    /// Key cache, shape = `num_kv_head x max_seq_len x head_dim`.
    k: Tensor<Storage>,
    /// Value cache, shape = `num_kv_head x max_seq_len x head_dim`.
    v: Tensor<Storage>,
}

impl<Storage> LayerCache<Storage> {
    /// Alloc KV Cache for all layers.
    pub fn new_layers(
        model: &dyn Llama2,
        tensor: impl Fn(DataType, &[udim]) -> Tensor<Storage>,
    ) -> Vec<Self> {
        let nkvh = model.num_key_value_heads() as udim;
        let max_seq_len = model.max_position_embeddings() as udim;
        let dh = (model.hidden_size() / model.num_attention_heads()) as udim;

        let dt = model.data_type();
        let shape = &[nkvh, max_seq_len, dh];

        (0..model.num_hidden_layers())
            .map(|_| Self {
                k: tensor(dt, shape),
                v: tensor(dt, shape),
            })
            .collect()
    }

    /// Get mutable references to the key and value cache.
    #[inline]
    pub fn get(&mut self) -> (&mut Tensor<Storage>, &mut Tensor<Storage>) {
        (&mut self.k, &mut self.v)
    }
}
