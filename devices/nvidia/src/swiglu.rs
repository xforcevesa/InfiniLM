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

pub struct Swiglu {
    ptx: Ptx,
    f: CString,
    block_size: c_uint,
}

impl PtxWapper for Swiglu {
    #[inline]
    fn ptx(&self) -> &Ptx {
        &self.ptx
    }
}

impl Swiglu {
    pub fn new(data_type: CudaDataType, cc: ComputeCapability, block_size: usize) -> Self {
        let ty_arg = data_type.name();
        let name = format!("swiglu_{ty_arg}");

        const SWIGLU: &str = include_str!("swiglu.cuh");
        let code = format!(
            r#"{SWIGLU}

extern "C" __global__ void {name}(
    {ty_arg} *__restrict__ gate,
    int const stride_gate,
    {ty_arg} const *__restrict__ up,
    int const stride_up
){{
    swiglu(gate, stride_gate, up, stride_up);
}}
"#
        );

        let (ptx, log) = Ptx::compile(code, cc);
        if !log.is_empty() {
            warn!("{log}");
        }
        Self {
            ptx: ptx.unwrap(),
            f: CString::new(name).unwrap(),
            block_size: block_size as _,
        }
    }

    pub fn launch<T, U>(
        &self,
        module: &ModuleSpore,
        gate: &mut Tensor<T>,
        up: &Tensor<U>,
        stream: &Stream,
    ) where
        T: DerefMut<Target = [DevByte]>,
        U: Deref<Target = [DevByte]>,
    {
        assert_eq!(gate.data_type(), up.data_type());
        assert_eq!(gate.shape(), up.shape());

        let &[seq_len, di] = gate.shape() else {
            panic!("gate shape: {:?}", gate.shape());
        };
        assert_eq!(gate.strides()[1], 1);
        assert_eq!(up.strides()[1], 1);

        let gate_ptr = (gate.physical().as_ptr() as isize + gate.bytes_offset()) as CUdeviceptr;
        let up_ptr = (up.physical().as_ptr() as isize + up.bytes_offset()) as CUdeviceptr;
        let params: [*const c_void; 4] = [
            (&gate_ptr) as *const _ as _,
            (&gate.strides()[0]) as *const _ as _,
            (&up_ptr) as *const _ as _,
            (&up.strides()[0]) as *const _ as _,
        ];

        #[inline]
        fn gcd(mut a: udim, mut b: udim) -> u32 {
            while b != 0 {
                let rem = a % b;
                a = b;
                b = rem;
            }
            a
        }

        let block_dims = gcd(self.block_size, di);
        let grid_dims = (seq_len, di / block_dims);

        let module = unsafe { module.sprout(stream.ctx()) };
        let kernel = module.get_kernel(&self.f);
        kernel.launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}
