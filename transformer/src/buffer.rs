use crate::{Llama2, Request};
use tensor::Tensor;

pub struct LayerBuffer<Storage> {
    pub qkv: Tensor<Storage>,
    pub gate_up: Tensor<Storage>,
    pub q_buf: Storage,
    pub att_buf: Storage,
}

impl<Storage> LayerBuffer<Storage> {
    /// 预分配逐层推理中用到的空间。
    pub fn alloc<T, U>(
        model: &dyn Llama2,
        requests: &[Request<T, U>],
        mut allocator: impl FnMut(usize) -> Storage,
    ) -> Self {
        // `nt` for number of tokens
        let (nt, max_seq_len, max_att_len) =
            requests
                .iter()
                .fold((0, 0, 0), |(nt, max_seq, max_att), r| {
                    let seq = r.seq_len() as usize;
                    let att = r.att_len() as usize;
                    (nt + seq, max_seq.max(seq), max_att.max(att))
                });

        let d = model.hidden_size();
        let nh = model.num_attention_heads();
        let nkvh = model.num_key_value_heads();
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = model.intermediate_size();
        let dt = model.data_type();

        Self {
            qkv: Tensor::new(
                dt,
                &[nt as _, (d + dkv + dkv) as _],
                allocator(nt * (d + dkv + dkv) * dt.size()),
            ),
            gate_up: Tensor::new(
                dt,
                &[nt as _, (di + di) as _],
                allocator(nt * (di + di) * dt.size()),
            ),
            q_buf: allocator(nh * max_seq_len * dh * dt.size()),
            att_buf: allocator(nh * max_seq_len * max_att_len * dt.size()),
        }
    }
}
