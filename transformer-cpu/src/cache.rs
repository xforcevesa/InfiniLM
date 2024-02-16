use model_parameters::Llama2;

pub struct LayerCache(Vec<u8>);

impl LayerCache {
    pub fn new_layers(model: &dyn Llama2) -> Vec<Self> {
        let dkv = model.num_key_value_heads();
        let ds = model.max_position_embeddings();
        let dh = model.hidden_size() / model.num_attention_heads();
        (0..model.num_hidden_layers())
            .map(|_| Self(vec![0; 2 * dkv * ds * dh]))
            .collect()
    }

    #[inline]
    pub fn get(&mut self) -> (&mut [u8], &mut [u8]) {
        let mid = self.0.len() / 2;
        self.0.split_at_mut(mid)
    }
}
