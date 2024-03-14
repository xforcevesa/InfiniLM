use crate::Storage;
use cuda::Stream;
use tensor::Tensor;
use transformer::Llama2;

pub(crate) struct ModelParameters<'ctx> {
    pub(crate) model_norm: Tensor<Storage<'ctx>>,
    pub(crate) lm_head: Tensor<Storage<'ctx>>,
    pub(crate) sync_event: cuda::Event<'ctx>,
}

impl<'ctx> ModelParameters<'ctx> {
    pub fn new(host: &dyn Llama2, stream: &'ctx Stream) -> Self {
        macro_rules! map {
            ($param:ident) => {
                unsafe {
                    host.$param()
                        .as_ref()
                        .map_physical(|slice| stream.from_host(slice).into())
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

pub(crate) struct LayersParameters<'ctx> {
    layers: Vec<LayerParameter<'ctx>>,
    current: usize,
}

impl<'ctx> LayersParameters<'ctx> {
    pub fn new(load_layers: usize, host: &dyn Llama2, stream: &Stream<'ctx>) -> Self {
        Self {
            layers: (0..host.num_hidden_layers().min(load_layers))
                .map(|layer| LayerParameter::new(host, layer, stream))
                .collect(),
            current: 0,
        }
    }

    #[inline]
    pub fn load(&mut self, layer: usize, host: &dyn Llama2, stream: &Stream<'ctx>) {
        let step = self.layers.len() - 1;
        let i = (self.current + step) % self.layers.len();
        let layer = (layer + step) % host.num_hidden_layers();
        self.layers[i].load(host, layer, stream);
    }

    #[inline]
    pub fn sync(&mut self, layer: usize, stream: &Stream<'ctx>) -> &LayerParameter<'ctx> {
        let i = self.current;
        self.current = (i + 1) % self.layers.len();

        let params = &self.layers[i];
        assert_eq!(params.layer, layer);
        stream.wait_for(&params.sync_event);

        params
    }
}

pub(crate) struct LayerParameter<'ctx> {
    pub input_layernorm: Tensor<Storage<'ctx>>,
    pub w_qkv: Tensor<Storage<'ctx>>,
    pub self_attn_o_proj: Tensor<Storage<'ctx>>,
    pub post_attention_layernorm: Tensor<Storage<'ctx>>,
    pub mlp_gate_up: Tensor<Storage<'ctx>>,
    pub mlp_down: Tensor<Storage<'ctx>>,

    layer: usize,
    sync_event: cuda::Event<'ctx>,
}

impl<'ctx> LayerParameter<'ctx> {
    pub fn new(host: &dyn Llama2, layer: usize, stream: &Stream<'ctx>) -> Self {
        macro_rules! map {
            ($param:ident) => {
                unsafe {
                    host.$param(layer)
                        .as_ref()
                        .map_physical(|slice| stream.from_host(slice).into())
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

    pub fn load(&mut self, host: &dyn Llama2, layer: usize, stream: &Stream<'ctx>) {
        if self.layer == layer {
            return;
        }

        macro_rules! update {
            ($param:ident) => {
                self.$param
                    .access_mut()
                    .physical_mut()
                    .copy_in_async(host.$param(layer).as_slice(), stream)
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
