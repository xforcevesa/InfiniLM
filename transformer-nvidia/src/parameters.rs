use crate::DevMem;
use cuda::Stream;
use model_parameters::Llama2;

pub(crate) struct ModelParameters<'a> {
    model_norm: DevMem<'a>,
    lm_head: DevMem<'a>,
    sync_event: cuda::Event,
}

impl<'a> ModelParameters<'a> {
    pub fn new(host: &dyn Llama2, stream: &'a Stream) -> Self {
        Self {
            model_norm: DevMem::from_slice(host.model_norm().as_slice(), stream),
            lm_head: DevMem::from_slice(host.lm_head().as_slice(), stream),
            sync_event: stream.record(),
        }
    }
}

impl Drop for ModelParameters<'_> {
    #[inline]
    fn drop(&mut self) {
        self.sync_event.synchronize();
    }
}

pub(crate) struct LayersParameters<'a>(Vec<LayerParameter<'a>>);

impl<'a> LayersParameters<'a> {
    pub fn new(load_layers: usize, host: &dyn Llama2, stream: &'a Stream) -> Self {
        Self(
            (0..host.num_hidden_layers().min(load_layers))
                .map(|layer| LayerParameter::new(host, layer, stream))
                .collect(),
        )
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn load(&mut self, layer: usize, model: &dyn Llama2, stream: &Stream) {
        let i = layer % self.0.len();
        self.0[i].load(model, layer, stream)
    }

    #[inline]
    pub fn sync(&self, layer: usize, stream: &Stream) -> &LayerParameter {
        let len = self.0.len();
        let params = &self.0[layer % len];
        assert_eq!(params.layer, layer);
        stream.wait_for(&params.sync_event);
        params
    }
}

pub(crate) struct LayerParameter<'a> {
    pub input_layernorm: DevMem<'a>,
    pub w_qkv: DevMem<'a>,
    pub self_attn_o_proj: DevMem<'a>,
    pub post_attention_layernorm: DevMem<'a>,
    pub mlp_gate_up: DevMem<'a>,
    pub mlp_down: DevMem<'a>,

    layer: usize,
    sync_event: cuda::Event,
}

impl<'a> LayerParameter<'a> {
    pub fn new(model: &dyn Llama2, layer: usize, stream: &'a Stream) -> Self {
        Self {
            input_layernorm: DevMem::from_slice(model.input_layernorm(layer).as_slice(), stream),
            w_qkv: DevMem::from_slice(model.w_qkv(layer).as_slice(), stream),
            self_attn_o_proj: DevMem::from_slice(model.self_attn_o_proj(layer).as_slice(), stream),
            post_attention_layernorm: DevMem::from_slice(
                model.post_attention_layernorm(layer).as_slice(),
                stream,
            ),
            mlp_gate_up: DevMem::from_slice(model.mlp_gate_up(layer).as_slice(), stream),
            mlp_down: DevMem::from_slice(model.mlp_down(layer).as_slice(), stream),
            layer,
            sync_event: stream.record(),
        }
    }

    pub fn load(&mut self, model: &dyn Llama2, layer: usize, stream: &Stream) {
        if self.layer == layer {
            return;
        }
        self.input_layernorm
            .copy_in(model.input_layernorm(layer).as_slice(), stream);
        self.w_qkv.copy_in(model.w_qkv(layer).as_slice(), stream);
        self.self_attn_o_proj
            .copy_in(model.self_attn_o_proj(layer).as_slice(), stream);
        self.post_attention_layernorm
            .copy_in(model.post_attention_layernorm(layer).as_slice(), stream);
        self.mlp_gate_up
            .copy_in(model.mlp_gate_up(layer).as_slice(), stream);
        self.mlp_down
            .copy_in(model.mlp_down(layer).as_slice(), stream);

        self.layer = layer;
        self.sync_event = stream.record();
    }
}

impl Drop for LayerParameter<'_> {
    #[inline]
    fn drop(&mut self) {
        self.sync_event.synchronize();
    }
}
