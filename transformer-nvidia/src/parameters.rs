use crate::DevMem;
use cuda::Stream;
use model_parameters::Llama2;
use tensor::Tensor;

pub(crate) struct ModelParameters<'a> {
    pub(crate) model_norm: Tensor<DevMem<'a>>,
    pub(crate) lm_head: Tensor<DevMem<'a>>,
    pub(crate) sync_event: cuda::Event,
}

impl<'a> ModelParameters<'a> {
    pub fn new(host: &dyn Llama2, stream: &'a Stream) -> Self {
        macro_rules! map {
            ($param:ident) => {
                unsafe {
                    host.$param()
                        .map_physical(|slice| DevMem::from_slice(slice, stream))
                }
            };
        }
        Self {
            model_norm: map!(model_norm),
            lm_head: map!(lm_head),
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

pub(crate) struct LayersParameters<'a> {
    layers: Vec<LayerParameter<'a>>,
    current: usize,
}

impl<'a> LayersParameters<'a> {
    pub fn new(load_layers: usize, host: &dyn Llama2, stream: &'a Stream) -> Self {
        Self {
            layers: (0..host.num_hidden_layers().min(load_layers))
                .map(|layer| LayerParameter::new(host, layer, stream))
                .collect(),
            current: 0,
        }
    }

    #[inline]
    pub fn load(&mut self, layer: usize, host: &dyn Llama2, stream: &Stream) {
        let step = self.layers.len() - 1;
        let i = (self.current + step) % self.layers.len();
        let layer = (layer + step) % host.num_hidden_layers();
        self.layers[i].load(host, layer, stream);
    }

    #[inline]
    pub fn sync(&mut self, layer: usize, stream: &Stream) -> &LayerParameter {
        let i = self.current;
        self.current = (i + 1) % self.layers.len();

        let params = &self.layers[i];
        assert_eq!(params.layer, layer);
        stream.wait_for(&params.sync_event);

        params
    }
}

pub(crate) struct LayerParameter<'a> {
    pub input_layernorm: Tensor<DevMem<'a>>,
    pub w_qkv: Tensor<DevMem<'a>>,
    pub self_attn_o_proj: Tensor<DevMem<'a>>,
    pub post_attention_layernorm: Tensor<DevMem<'a>>,
    pub mlp_gate_up: Tensor<DevMem<'a>>,
    pub mlp_down: Tensor<DevMem<'a>>,

    layer: usize,
    sync_event: cuda::Event,
}

impl<'a> LayerParameter<'a> {
    pub fn new(host: &dyn Llama2, layer: usize, stream: &'a Stream) -> Self {
        macro_rules! map {
            ($param:ident) => {
                unsafe {
                    host.$param(layer)
                        .map_physical(|slice| DevMem::from_slice(slice, stream))
                }
            };
        }
        Self {
            input_layernorm: map!(input_layernorm),
            w_qkv: map!(w_qkv),
            self_attn_o_proj: map!(self_attn_o_proj),
            post_attention_layernorm: map!(post_attention_layernorm),
            mlp_gate_up: map!(mlp_gate_up),
            mlp_down: map!(mlp_down),
            layer,
            sync_event: stream.record(),
        }
    }

    pub fn load(&mut self, host: &dyn Llama2, layer: usize, stream: &Stream) {
        if self.layer == layer {
            return;
        }

        macro_rules! update {
            ($param:ident) => {
                self.$param
                    .physical_mut()
                    .copy_in(host.$param(layer).as_slice(), stream)
            };
        }
        update!(input_layernorm);
        update!(w_qkv);
        update!(self_attn_o_proj);
        update!(post_attention_layernorm);
        update!(mlp_gate_up);
        update!(mlp_down);

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
