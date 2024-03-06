use cuda::{bindings::CUdeviceptr, AsRaw, ContextGuard, CudaDataType, DevMem, Module, Ptx, Stream};
use std::{
    ffi::{c_uint, c_void, CString},
    ops::Deref,
};
use tensor::Tensor;

pub struct FusedSoftmax<'ctx> {
    module: Module<'ctx>,
    padding: CString,
    folding: CString,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl<'ctx> FusedSoftmax<'ctx> {
    pub fn new(
        data_type: CudaDataType,
        max_seq_len: usize,
        block_size: usize,
        ctx: &'ctx ContextGuard<'ctx>,
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

        let (ptx, log) = Ptx::compile(code);
        if !log.is_empty() {
            warn!("{log}");
        }
        Self {
            module: ctx.load(&ptx.unwrap()),
            padding: CString::new(padding).unwrap(),
            folding: CString::new(folding).unwrap(),
            block_size: block_size as _,
            items_per_thread: items_per_thread as _,
        }
    }
}

impl FusedSoftmax<'_> {
    pub fn launch<'a, T>(&self, att: &Tensor<T>, stream: &Stream)
    where
        T: Deref<Target = DevMem<'a>>,
    {
        assert!(att.is_contiguous());
        let &[nh, seq_len, att_len] = att.shape() else {
            panic!("Invalid attention shape");
        };
        let &[stride_y, stride_x, 1] = att.strides() else {
            unreachable!();
        };

        let grid_dims = (nh, seq_len);
        let (name, block_dims) = if att_len <= self.block_size {
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

        let kernel = self.module.get_kernel(name);
        kernel.launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}
