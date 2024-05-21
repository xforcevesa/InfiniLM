use crate::PtxWapper;
use cuda::{
    bindings::CUdeviceptr, ComputeCapability, ContextSpore, CudaDataType, DevByte, ModuleSpore,
    Ptx, Stream,
};
use std::{
    ffi::{c_uint, c_void, CString},
    ops::{Deref, DerefMut},
};
use tensor::{udim, Tensor};

pub struct RmsNormalization {
    ptx: Ptx,
    padding: CString,
    folding: CString,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl PtxWapper for RmsNormalization {
    #[inline]
    fn ptx(&self) -> &Ptx {
        &self.ptx
    }
}

impl RmsNormalization {
    pub fn new(
        data_type: CudaDataType,
        max_item_size: usize,
        cc: ComputeCapability,
        block_size: usize,
    ) -> Self {
        let ty_arg = data_type.name();
        let items_per_thread = (max_item_size + block_size - 1) / block_size;
        let padding = format!("rms_normalization_padding_{block_size}");
        let folding = format!("rms_normalization_folding_{items_per_thread}x{block_size}");

        const RMS_NORMALIZATION: &str = include_str!("rms_norm.cuh");
        let code = format!(
            r#"{RMS_NORMALIZATION}

extern "C" __global__ void {padding}(
    {ty_arg}       *__restrict__ o,
    unsigned int const    stride_o,
    {ty_arg} const *__restrict__ x,
    unsigned int const    stride_x,
    {ty_arg} const *__restrict__ w,
    float epsilon
){{
    padding<{block_size}>
    (o, stride_o, x, stride_x, w, epsilon);
}}

extern "C" __global__ void {folding}(
    {ty_arg}       *__restrict__ o,
    unsigned int const    stride_o,
    {ty_arg} const *__restrict__ x,
    unsigned int const    stride_x,
    {ty_arg} const *__restrict__ w,
    float epsilon,
    unsigned int const items_size
){{
    folding<{block_size}, {items_per_thread}>
    (o, stride_o, x, stride_x, w, epsilon, items_size);
}}
"#
        );

        let (ptx, log) = Ptx::compile(code, cc);
        if !log.is_empty() {
            warn!("{log}");
        }
        Self {
            ptx: ptx.unwrap(),
            padding: CString::new(padding).unwrap(),
            folding: CString::new(folding).unwrap(),
            block_size: block_size as _,
            items_per_thread: items_per_thread as _,
        }
    }

    pub fn launch<T, U, V>(
        &self,
        module: &ModuleSpore,
        o: &mut Tensor<T>,
        x: &Tensor<U>,
        w: &Tensor<V>,
        epsilon: f32,
        stream: &Stream,
    ) where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
        V: Deref<Target = [DevByte]>,
    {
        let &[n, d] = o.shape() else { panic!() };
        let dt = o.data_type();

        assert_eq!(x.data_type(), dt);
        assert_eq!(w.data_type(), dt);
        assert_eq!(o.shape(), x.shape());
        assert_eq!(w.shape(), &[d]);
        assert!(o.contiguous_len() >= 1);
        assert!(x.contiguous_len() >= 1);
        assert!(w.is_contiguous());

        let o_ptr = (o.physical().as_ptr() as isize + o.bytes_offset()) as CUdeviceptr;
        let x_ptr = (x.physical().as_ptr() as isize + x.bytes_offset()) as CUdeviceptr;
        let w_ptr = (w.physical().as_ptr() as isize + w.bytes_offset()) as CUdeviceptr;
        let stride_o = o.strides()[0] as usize;
        let stride_x = x.strides()[0] as usize;
        let items_len = d as udim;
        let params: [*const c_void; 7] = [
            (&o_ptr) as *const _ as _,
            (&stride_o) as *const _ as _,
            (&x_ptr) as *const _ as _,
            (&stride_x) as *const _ as _,
            (&w_ptr) as *const _ as _,
            (&epsilon) as *const _ as _,
            (&items_len) as *const _ as _,
        ];
        let module = unsafe { module.sprout(stream.ctx()) };
        if items_len <= self.block_size {
            let kernel = module.get_kernel(&self.padding);
            kernel.launch(n, items_len, params.as_ptr(), 0, Some(stream));
        } else {
            let block_size = (items_len + self.items_per_thread - 1) / self.items_per_thread;
            let kernel = module.get_kernel(&self.folding);
            kernel.launch(n, block_size, params.as_ptr(), 0, Some(stream));
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
    dev.context()
        .apply(|_| RmsNormalization::new(CudaDataType::f16, 2048, dev.compute_capability(), 1024));
}
