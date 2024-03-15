use cuda::{
    bindings::CUdeviceptr, AsRaw, ContextGuard, CudaDataType, DevSlice, Module, Ptx, Stream,
};
use std::{
    ffi::{c_uint, c_void, CString},
    ops::{Deref, DerefMut},
};
use tensor::{udim, Tensor};

pub struct RmsNormalization<'ctx> {
    module: Module<'ctx>,
    padding: CString,
    folding: CString,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl<'ctx> RmsNormalization<'ctx> {
    pub fn new(
        data_type: CudaDataType,
        max_item_size: usize,
        block_size: usize,
        ctx: &'ctx ContextGuard<'ctx>,
    ) -> Self {
        let ty_arg = data_type.name();
        let items_per_thread = (max_item_size + block_size - 1) / block_size;
        let padding = format!("rms_normalization_padding_{block_size}");
        let folding = format!("rms_normalization_folding_{items_per_thread}x{block_size}");

        const RMS_NORMALIZATION: &str = include_str!("rms_norm.cuh");
        let code = format!(
            r#"{RMS_NORMALIZATION}

extern "C" __global__ void {padding}(
    {ty_arg}       *__restrict__ y,
    {ty_arg} const *__restrict__ x,
    {ty_arg} const *__restrict__ w,
    float epsilon,
    unsigned int const leading_dim
){{
    padding<{block_size}>
    (y, x, w, epsilon, leading_dim);
}}

extern "C" __global__ void {folding}(
    {ty_arg}       *__restrict__ y,
    {ty_arg} const *__restrict__ x,
    {ty_arg} const *__restrict__ w,
    float epsilon,
    unsigned int const leading_dim,
    unsigned int const items_size
){{
    folding<{block_size}, {items_per_thread}>
    (y, x, w, epsilon, leading_dim, items_size);
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

impl RmsNormalization<'_> {
    pub fn launch<T, U, V>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<U>,
        w: &Tensor<V>,
        epsilon: f32,
        stream: &Stream,
    ) where
        T: DerefMut<Target = DevSlice>,
        U: Deref<Target = DevSlice>,
        V: Deref<Target = DevSlice>,
    {
        debug_assert_eq!(x.shape(), y.shape());
        let &[row, col] = x.shape() else { panic!() };
        debug_assert_eq!(&[col], w.shape());

        let y_ptr = (unsafe { y.physical().as_raw() } as isize + y.bytes_offset()) as CUdeviceptr;
        let x_ptr = (unsafe { x.physical().as_raw() } as isize + x.bytes_offset()) as CUdeviceptr;
        let w_ptr = (unsafe { w.physical().as_raw() } as isize + w.bytes_offset()) as CUdeviceptr;
        let leading_dim = x.strides()[0] as udim;
        let items_len = col as udim;
        let params: [*const c_void; 6] = [
            (&y_ptr) as *const _ as _,
            (&x_ptr) as *const _ as _,
            (&w_ptr) as *const _ as _,
            (&epsilon) as *const _ as _,
            (&leading_dim) as *const _ as _,
            (&items_len) as *const _ as _,
        ];
        if items_len <= self.block_size {
            let kernel = self.module.get_kernel(&self.padding);
            kernel.launch(row, items_len, params.as_ptr(), 0, Some(stream));
        } else {
            let block_size = (items_len + self.items_per_thread - 1) / self.items_per_thread;
            let kernel = self.module.get_kernel(&self.folding);
            kernel.launch(row, block_size, params.as_ptr(), 0, Some(stream));
        }
    }
}

#[test]
fn test_kernel() {
    use cuda::CudaDataType;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        RmsNormalization::new(CudaDataType::half, 2048, 1024, ctx);
    });
}
