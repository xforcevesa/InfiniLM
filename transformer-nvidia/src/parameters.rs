use crate::Storage;
use cuda::{ContextResource, ContextSpore, DevMem, DevMemSpore, EventSpore, Stream};
use std::rc::Rc;
use tensor::Tensor;
use transformer::Llama2;

pub(crate) struct ModelParameters {
    model_norm: Tensor<Rc<DevMemSpore>>,
    lm_head: Tensor<Rc<DevMemSpore>>,
    sync_event: EventSpore,
}

impl ModelParameters {
    pub fn new(host: &dyn Llama2, stream: &Stream) -> Self {
        macro_rules! map {
            ($param:ident) => {
                unsafe {
                    host.$param()
                        .as_ref()
                        .map_physical(|slice| Rc::new(stream.from_host(slice).sporulate()))
                }
            };
        }
        Self {
            model_norm: map!(model_norm),
            lm_head: map!(lm_head).transpose(&[1, 0]),
            sync_event: stream.record().sporulate(),
        }
    }

    pub unsafe fn release<'ctx>(
        &self,
        stream: &Stream<'ctx>,
    ) -> (Tensor<DevMem<'ctx>>, Tensor<DevMem<'ctx>>) {
        let ctx = stream.ctx();
        stream.wait_for(&self.sync_event.sprout(ctx));
        (
            self.model_norm.clone().map_physical(|s| s.sprout(ctx)),
            self.lm_head.clone().map_physical(|s| s.sprout(ctx)),
        )
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
            w_qkv: map!(w_qkv).transpose(&[1, 0]),
            self_attn_o_proj: map!(self_attn_o_proj).transpose(&[1, 0]),
            post_attention_layernorm: map!(post_attention_layernorm),
            mlp_gate_up: map!(mlp_gate_up).transpose(&[1, 0]),
            mlp_down: map!(mlp_down).transpose(&[1, 0]),
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
