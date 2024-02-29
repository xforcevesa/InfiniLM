use crate::storage::DevMem;
use cuda::{bindings::CUdeviceptr, AsRaw, ContextGuard, CudaDataType, KernelFn, Stream};
use std::ffi::{c_uint, c_void};
use tensor::{udim, Tensor};

pub struct Swiglu {
    f: KernelFn,
    block_size: c_uint,
}

impl Swiglu {
    pub fn new(data_type: CudaDataType, block_size: usize, ctx: &ContextGuard) -> Self {
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

        ctx.compile(code);
        Self {
            f: KernelFn::get(name).unwrap(),
            block_size: block_size as _,
        }
    }

    pub fn launch(&self, gate: &Tensor<DevMem>, up: &Tensor<DevMem>, stream: &Stream) {
        assert_eq!(gate.data_type(), up.data_type());
        assert_eq!(gate.shape(), up.shape());

        let &[seq_len, di] = gate.shape() else {
            panic!("gate shape: {:?}", gate.shape());
        };
        assert_eq!(gate.strides()[1], 1);
        assert_eq!(up.strides()[1], 1);

        let gate_ptr =
            (unsafe { gate.physical().as_raw() } as isize + gate.bytes_offset()) as CUdeviceptr;
        let up_ptr =
            (unsafe { up.physical().as_raw() } as isize + up.bytes_offset()) as CUdeviceptr;
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

        let block_dims = gcd(di, self.block_size);
        let grid_dims = (di / block_dims, seq_len);
        self.f
            .launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}
