use crate::{memory::Layer, ConfigJson, Llama2, Memory, Storage};
use std::{ops::Deref, ptr::NonNull, slice::from_raw_parts_mut, sync::Arc};
use tensor::Tensor;

pub trait Allocator {
    unsafe fn allocate(&self, size: usize) -> NonNull<u8>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>);
}

struct TotalStorage<A: Allocator> {
    ptr: NonNull<u8>,
    len: usize,
    allocator: A,
}

impl<A: Allocator> Deref for TotalStorage<A> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<A: Allocator> Drop for TotalStorage<A> {
    fn drop(&mut self) {
        unsafe { self.allocator.deallocate(self.ptr) }
    }
}

impl Memory {
    pub fn realloc_with(src: &dyn Llama2, allocator: impl Allocator + 'static) -> Self {
        let len = src.size();
        let ptr = unsafe { allocator.allocate(len) };
        let total = Arc::new(TotalStorage {
            ptr,
            len,
            allocator,
        });

        struct Writer<A: Allocator> {
            total: Arc<TotalStorage<A>>,
            offset: usize,
        }
        impl<A: Allocator + 'static> Writer<A> {
            fn write(&mut self, tensor: Tensor<Storage>) -> Tensor<Storage> {
                let offset = self.offset;
                let ptr = self.total.ptr.as_ptr();
                let len = tensor.bytes_size();
                self.offset += len;
                unsafe { tensor.reform_to_raw(from_raw_parts_mut(ptr.add(offset), len)) };
                Tensor::new(
                    tensor.data_type(),
                    tensor.shape(),
                    Storage::new(self.total.clone(), offset, len),
                )
            }
        }

        let mut writer = Writer { total, offset: 0 };
        Self {
            config: ConfigJson::from(src),
            embed_tokens: writer.write(src.embed_tokens()),
            layers: (0..src.num_hidden_layers())
                .map(|layer| Layer {
                    input_layernorm: writer.write(src.input_layernorm(layer)),
                    w_qkv: writer.write(src.w_qkv(layer)),
                    self_attn_o_proj: writer.write(src.self_attn_o_proj(layer)),
                    post_attention_layernorm: writer.write(src.post_attention_layernorm(layer)),
                    mlp_gate_up: writer.write(src.mlp_gate(layer)),
                    mlp_down: writer.write(src.mlp_down(layer)),
                })
                .collect(),
            model_norm: writer.write(src.model_norm()),
            lm_head: writer.write(src.lm_head()),
        }
    }
}
