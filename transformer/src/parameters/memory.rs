use super::{ConfigJson, DataType, HostMemory, Llama2};
use common::utok;
use tensor::Tensor;

pub struct Memory<'a> {
    pub(super) config: ConfigJson,
    pub(super) embed_tokens: Tensor<HostMemory<'a>>,
    pub(super) layers: Vec<Layer<'a>>,
    pub(super) model_norm: Tensor<HostMemory<'a>>,
    pub(super) lm_head: Tensor<HostMemory<'a>>,
}

pub(super) struct Layer<'a> {
    pub input_layernorm: Tensor<HostMemory<'a>>,
    pub w_qkv: Tensor<HostMemory<'a>>,
    pub self_attn_o_proj: Tensor<HostMemory<'a>>,
    pub post_attention_layernorm: Tensor<HostMemory<'a>>,
    pub mlp_gate_up: Tensor<HostMemory<'a>>,
    pub mlp_down: Tensor<HostMemory<'a>>,
}

impl<'a> Llama2 for Memory<'a> {
    #[inline]
    fn bos_token_id(&self) -> utok {
        self.config.bos_token_id
    }

    #[inline]
    fn eos_token_id(&self) -> utok {
        self.config.eos_token_id
    }

    #[inline]
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[inline]
    fn intermediate_size(&self) -> usize {
        self.config.intermediate_size
    }

    #[inline]
    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    #[inline]
    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    #[inline]
    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    #[inline]
    fn num_key_value_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    #[inline]
    fn rms_norm_eps(&self) -> f32 {
        self.config.rms_norm_eps
    }

    #[inline]
    fn rope_theta(&self) -> f32 {
        self.config.rope_theta
    }

    #[inline]
    fn data_type(&self) -> DataType {
        self.config.torch_dtype
    }

    #[inline]
    fn embed_tokens(&self) -> Tensor<HostMemory> {
        self.embed_tokens.clone()
    }

    #[inline]
    fn input_layernorm(&self, layer: usize) -> Tensor<HostMemory> {
        self.layers[layer].input_layernorm.clone()
    }

    #[inline]
    fn w_qkv(&self, layer: usize) -> Tensor<HostMemory> {
        self.layers[layer].w_qkv.clone()
    }

    #[inline]
    fn self_attn_q_proj(&self, layer: usize) -> Tensor<HostMemory> {
        let d = self.config.hidden_size;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.end = physical.range.start + d * d * dt;
        Tensor::new(self.config.torch_dtype, &[d as _, d as _], physical)
    }

    #[inline]
    fn self_attn_k_proj(&self, layer: usize) -> Tensor<HostMemory> {
        let d = self.config.hidden_size;
        let dkv = self.kv_hidden_size();
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.start += d * d * dt;
        physical.range.end = physical.range.start + dkv * d * dt;
        Tensor::new(self.config.torch_dtype, &[dkv as _, d as _], physical)
    }

    #[inline]
    fn self_attn_v_proj(&self, layer: usize) -> Tensor<HostMemory> {
        let d = self.config.hidden_size;
        let dkv = self.kv_hidden_size();
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].w_qkv.physical().clone();
        physical.range.start += (d + dkv) * d * dt;
        physical.range.end = physical.range.start + dkv * d * dt;
        Tensor::new(self.config.torch_dtype, &[dkv as _, d as _], physical)
    }

    #[inline]
    fn self_attn_o_proj(&self, layer: usize) -> Tensor<HostMemory> {
        self.layers[layer].self_attn_o_proj.clone()
    }

    #[inline]
    fn post_attention_layernorm(&self, layer: usize) -> Tensor<HostMemory> {
        self.layers[layer].post_attention_layernorm.clone()
    }

    #[inline]
    fn mlp_gate_up(&self, layer: usize) -> Tensor<HostMemory> {
        self.layers[layer].mlp_gate_up.clone()
    }

    #[inline]
    fn mlp_gate(&self, layer: usize) -> Tensor<HostMemory> {
        let di = self.config.intermediate_size;
        let d = self.config.hidden_size;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].mlp_gate_up.physical().clone();
        physical.range.end = physical.range.start + di * d * dt;
        Tensor::new(self.config.torch_dtype, &[di as _, d as _], physical)
    }

    #[inline]
    fn mlp_down(&self, layer: usize) -> Tensor<HostMemory> {
        self.layers[layer].mlp_down.clone()
    }

    #[inline]
    fn mlp_up(&self, layer: usize) -> Tensor<HostMemory> {
        let di = self.config.intermediate_size;
        let d = self.config.hidden_size;
        let dt = self.config.torch_dtype.size();
        let mut physical = self.layers[layer].mlp_gate_up.physical().clone();
        physical.range.start += di * d * dt;
        physical.range.end = physical.range.start + di * d * dt;
        Tensor::new(self.config.torch_dtype, &[di as _, d as _], physical)
    }

    #[inline]
    fn model_norm(&self) -> Tensor<HostMemory> {
        self.model_norm.clone()
    }

    #[inline]
    fn lm_head(&self) -> Tensor<HostMemory> {
        self.lm_head.clone()
    }
}

#[test]
fn test_load() {
    use super::SafeTensorError;
    use std::{io::ErrorKind::NotFound, time::Instant};

    let t0 = Instant::now();
    let safetensors = Memory::load_safetensors_from_dir("../../TinyLlama-1.1B-Chat-v1.0");
    let t1 = Instant::now();
    println!("mmap {:?}", t1 - t0);

    let safetensors = match safetensors {
        Ok(m) => m,
        Err(SafeTensorError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let t0 = Instant::now();
    let _inside_memory = Memory::cast(&safetensors, DataType::F32);
    let t1 = Instant::now();
    println!("cast {:?}", t1 - t0);
}
