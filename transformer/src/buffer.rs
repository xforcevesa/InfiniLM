use crate::{Llama2, Request};
use tensor::{udim, LocalSplitable, Tensor};

pub struct LayerBuffer<Storage> {
    pub qkv: Tensor<LocalSplitable<Storage>>,
    pub gate_up: Tensor<LocalSplitable<Storage>>,
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

        let nt = nt as udim;
        let d = model.hidden_size() as udim;
        let nh = model.num_attention_heads();
        let nkvh = model.num_key_value_heads() as udim;
        let dh = d as usize / nh;
        let dkv = nkvh * dh as udim;
        let di = model.intermediate_size() as udim;
        let dt = model.data_type();

        Self {
            qkv: Tensor::alloc(dt, &[nt, d + dkv + dkv], |len| allocator(len).into()),
            gate_up: Tensor::alloc(dt, &[nt, di + di], |len| allocator(len).into()),
            q_buf: allocator(nh * max_seq_len * dh * dt.size()),
            att_buf: allocator(nh * max_seq_len * max_att_len * dt.size()),
        }
    }
}
