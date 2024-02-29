use crate::storage::DevMem;
use cuda::{bindings::CUdeviceptr, AsRaw, ContextGuard, CudaDataType, KernelFn, Stream};
use std::ffi::{c_uint, c_void};
use tensor::Tensor;

pub struct FusedSoftmax {
    padding: KernelFn,
    folding: KernelFn,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl FusedSoftmax {
    pub fn new(
        data_type: CudaDataType,
        max_seq_len: usize,
        block_size: usize,
        ctx: &ContextGuard,
    ) -> Self {
        let ty_arg = data_type.name();
        let mask = "AttentionCausualMask";
        let items_per_thread = (max_seq_len + block_size - 1) / block_size;
        let padding = format!("fused_softmax_padding_{block_size}");
        let folding = format!("fused_softmax_folding_{block_size}x{items_per_thread}");

        const FUSED_SOFTMAX: &str = include_str!("fused_softmax.cuh");
        let code = format!(
            r#"{FUSED_SOFTMAX}

extern "C" __global__ void {padding}(
    {ty_arg} *__restrict__ att,
    int const stride_x,
    int const stride_y,
    int const stride_z
){{
    padding<{block_size}>
    (att, {mask}(), stride_x, stride_y, stride_z);
}}

extern "C" __global__ void {folding}(
    {ty_arg} *__restrict__ att,
    unsigned int const stride_x,
             int const stride_y,
             int const stride_z,
             int const att_len
){{
    folding<{block_size}, {items_per_thread}>
    (att, {mask}(), att_len, stride_x, stride_y, stride_z);
}}
"#
        );

        ctx.compile(code);
        Self {
            padding: KernelFn::get(padding).unwrap(),
            folding: KernelFn::get(folding).unwrap(),
            block_size: block_size as _,
            items_per_thread: items_per_thread as _,
        }
    }

    pub fn launch(&self, att: &Tensor<DevMem>, stream: &Stream) {
        assert!(att.is_contiguous());
        let &[nh, seq_len, att_len] = att.shape() else {
            panic!("Invalid attention shape");
        };
        let &[stride_y, stride_x, 1] = att.strides() else {
            unreachable!();
        };
        println!("nh = {nh}, seq_len = {seq_len}, att_len = {att_len}, stride_x = {stride_x}, stride_y = {stride_y}");

        let grid_dims = (seq_len, nh);
        let (kernel, block_dims) = if att_len <= self.block_size {
            (&self.padding, att_len)
        } else {
            (
                &self.folding,
                (att_len + self.items_per_thread - 1) / self.items_per_thread,
            )
        };

        let ptr = (unsafe { att.physical().as_raw() } as isize + att.bytes_offset()) as CUdeviceptr;
        let params: [*const c_void; 5] = [
            (&ptr) as *const _ as _,
            (&stride_x) as *const _ as _,
            (&stride_y) as *const _ as _,
            (&0u32) as *const _ as _,
            (&att_len) as *const _ as _,
        ];

        kernel.launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}
