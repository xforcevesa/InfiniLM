use cuda::Stream;
use model_parameters::Llama2;

pub(crate) struct ModelParameters {
    model_norm: cuda::DevBlob,
    lm_head: cuda::DevBlob,
    sync_event: cuda::Event,
}

impl ModelParameters {
    pub fn new(host: &dyn Llama2, stream: &Stream) -> Self {
        Self {
            model_norm: stream.from_slice(host.model_norm().as_slice()),
            lm_head: stream.from_slice(host.lm_head().as_slice()),
            sync_event: stream.record(),
        }
    }

    #[inline]
    pub fn sync(&self) {
        self.sync_event.synchronize();
    }
}

impl Drop for ModelParameters {
    #[inline]
    fn drop(&mut self) {
        self.sync_event.synchronize();
    }
}

pub(crate) struct LayersParameters(Vec<LayerParameter>);

impl LayersParameters {
    pub fn new(load_layers: usize, host: &dyn Llama2, stream: &Stream) -> Self {
        Self(
            (0..host.num_hidden_layers().min(load_layers))
                .map(|layer| LayerParameter::load(host, layer, stream))
                .collect(),
        )
    }

    #[inline]
    pub fn sync(&self, layer: usize) {
        let params = self.0.get(layer % self.0.len()).unwrap();
        assert_eq!(params.layer, layer);
        params.sync_event.synchronize();
    }
}

struct LayerParameter {
    input_layernorm: cuda::DevBlob,
    w_qkv: cuda::DevBlob,
    self_attn_o_proj: cuda::DevBlob,
    post_attention_layernorm: cuda::DevBlob,
    mlp_gate_up: cuda::DevBlob,
    mlp_down: cuda::DevBlob,

    layer: usize,
    sync_event: cuda::Event,
}

impl LayerParameter {
    pub fn load(model: &dyn Llama2, layer: usize, stream: &Stream) -> Self {
        Self {
            input_layernorm: stream.from_slice(model.input_layernorm(layer).as_slice()),
            w_qkv: stream.from_slice(model.w_qkv(layer).as_slice()),
            self_attn_o_proj: stream.from_slice(model.self_attn_o_proj(layer).as_slice()),
            post_attention_layernorm: stream
                .from_slice(model.post_attention_layernorm(layer).as_slice()),
            mlp_gate_up: stream.from_slice(model.mlp_gate_up(layer).as_slice()),
            mlp_down: stream.from_slice(model.mlp_down(layer).as_slice()),
            layer,
            sync_event: stream.record(),
        }
    }
}

impl Drop for LayerParameter {
    #[inline]
    fn drop(&mut self) {
        self.sync_event.synchronize();
    }
}
