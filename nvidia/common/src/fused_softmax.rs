use crate::PtxWapper;
use cuda::{bindings::CUdeviceptr, ContextSpore, CudaDataType, DevByte, ModuleSpore, Ptx, Stream};
use std::{
    ffi::{c_uint, c_void, CString},
    ops::DerefMut,
};
use tensor::Tensor;

pub struct FusedSoftmax {
    ptx: Ptx,
    padding: CString,
    folding: CString,
    block_size: c_uint,
}

impl PtxWapper for FusedSoftmax {
    #[inline]
    fn ptx(&self) -> &Ptx {
        &self.ptx
    }
}

impl FusedSoftmax {
    pub fn new(data_type: CudaDataType, max_seq_len: usize, block_size: usize) -> Self {
        let ty_arg = data_type.name();
        let mask = "AttentionCausualMask";
        let max_items_per_thread = (max_seq_len + block_size - 1) / block_size;
        let padding = format!("fused_softmax_padding_{block_size}");
        let folding = format!("fused_softmax_folding_{block_size}x{max_items_per_thread}");

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
    folding<{block_size}, {max_items_per_thread}>
    (att, {mask}(), att_len, stride_x, stride_y, stride_z);
}}
"#
        );

        let (ptx, log) = Ptx::compile(code);
        if !log.is_empty() {
            warn!("{log}");
        }
        Self {
            ptx: ptx.unwrap(),
            padding: CString::new(padding).unwrap(),
            folding: CString::new(folding).unwrap(),
            block_size: block_size as _,
        }
    }

    pub fn launch<T>(&self, module: &ModuleSpore, att: &mut Tensor<T>, stream: &Stream)
    where
        T: DerefMut<Target = [DevByte]>,
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
            // FIXME: 极度怪异的行为。
            // 如果 block dims 不取 self.block_size, kernel 会在随机位置计算出错误数据。
            // 然而，如果打印 block dims，计算就不会出错。只能打印，写入带内存屏障的原子变量、锁、Flush 均无效。
            // 现在这样浪费了一些线程。
            // let mut block_dims = 0;
            // for items_per_thread in 2.. {
            //     block_dims = (att_len + items_per_thread - 1) / items_per_thread;
            //     block_dims = (block_dims + 31) / 32 * 32;
            //     if block_dims <= self.block_size {
            //         break;
            //     }
            // }
            (&self.folding, self.block_size)
        };
        // println!("block dims = {block_dims}");

        let ptr = (att.physical().as_ptr() as isize + att.bytes_offset()) as CUdeviceptr;
        let params: [*const c_void; 5] = [
            (&ptr) as *const _ as _,
            (&stride_x) as *const _ as _,
            (&stride_y) as *const _ as _,
            (&0u32) as *const _ as _,
            (&att_len) as *const _ as _,
        ];

        let module = unsafe { module.sprout(stream.ctx()) };
        let kernel = module.get_kernel(name);
        kernel.launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}

#[test]
fn test_kernel() {
    use cuda::{ContextResource, CudaDataType};
    use half::f16;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();

        const NH: usize = 1;
        const SEQ_LEN: usize = 768;
        const ATT_LEN: usize = 768;

        let data = vec![f16::from_f32(1.0); NH * SEQ_LEN * ATT_LEN];

        let mut att0 = Tensor::new(
            tensor::DataType::F16,
            &[NH as _, SEQ_LEN as _, ATT_LEN as _],
            stream.from_host(&data),
        );
        let mut att1 = Tensor::new(
            tensor::DataType::F16,
            &[NH as _, SEQ_LEN as _, ATT_LEN as _],
            stream.from_host(&data),
        );

        {
            let kernel = FusedSoftmax::new(CudaDataType::f16, 2048, 1024);
            let mut module = ctx.load(&kernel.ptx).sporulate();
            kernel.launch(&module, &mut att0, &stream);
            stream.synchronize();
            unsafe { module.kill(ctx) };
        }
        {
            let kernel = FusedSoftmax::new(CudaDataType::f16, 2048, 512);
            let mut module = ctx.load(&kernel.ptx).sporulate();
            kernel.launch(&module, &mut att1, &stream);
            stream.synchronize();
            unsafe { module.kill(ctx) };
        }

        let att0 = crate::map_tensor(&att0);
        let att1 = crate::map_tensor(&att1);
        assert!(att0.physical() == att1.physical());
    });
}
