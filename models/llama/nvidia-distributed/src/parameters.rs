use crate::distribute::{DistributeScheme, Distributer};
use common_nv::{
    cuda::{Context, ContextGuard, ContextResource, ContextSpore, DevByte, DevMem, DevMemSpore},
    udim, Tensor,
};
use std::time::Instant;

pub struct ParameterMatrix {
    scheme: DistributeScheme,
    matrix: Vec<DevMemSpore>,
}

impl ParameterMatrix {
    pub fn load(model: &llama::Storage, contexts: &[Context]) -> Self {
        let align = contexts
            .iter()
            .map(|ctx| ctx.device().alignment())
            .max()
            .unwrap();

        let nlayers = model.config.nlayers as usize;
        let mut matrix = Vec::with_capacity(contexts.len() * nlayers);

        let distributer = Distributer::new(model, contexts.len(), align);
        let time = Instant::now();
        for (i, context) in contexts.iter().enumerate() {
            context.apply(|ctx| {
                for layer in 0..nlayers {
                    matrix.push(ctx.from_host(&distributer.distribute(layer, i)).sporulate());
                }
            });
        }
        info!("distribute {:?}", time.elapsed());

        Self {
            scheme: distributer.scheme().clone(),
            matrix,
        }
    }

    pub unsafe fn kill(&mut self, contexts: &[Context]) {
        assert_eq!(contexts.len(), self.scheme.n);
        let nlayers = self.matrix.len() / self.scheme.n;
        for (i, context) in contexts.iter().enumerate() {
            context.apply(|ctx| {
                for element in &mut self.matrix[i * nlayers..][..nlayers] {
                    element.kill(ctx);
                }
            });
        }
    }
}

pub struct Layer<'ctx> {
    scheme: &'ctx DistributeScheme,
    mem: DevMem<'ctx>,
}

impl ParameterMatrix {
    pub fn get<'ctx>(&'ctx self, layer: usize, i: usize, ctx: &'ctx ContextGuard) -> Layer<'ctx> {
        let nlayers = self.matrix.len() / self.scheme.n;
        Layer {
            scheme: &self.scheme,
            mem: unsafe { self.matrix[i * nlayers + layer].sprout(ctx) },
        }
    }
}

impl Layer<'_> {
    #[inline]
    pub fn input_layernorm(&self) -> Tensor<&[DevByte]> {
        let d = self.scheme.nh * self.scheme.dh;
        Tensor::new(
            self.scheme.dt,
            &[d],
            &self.mem[self.scheme.input_layernorm..],
        )
    }

    #[inline]
    pub fn w_qkv(&self) -> Tensor<&[DevByte]> {
        let nh = self.scheme.nh;
        let nkvh = self.scheme.nkvh;
        let dh = self.scheme.dh;
        let d = nh * dh;
        let n = self.scheme.n as udim;
        Tensor::new(
            self.scheme.dt,
            &[d, (nh + nkvh + nkvh) / n * dh],
            &self.mem[self.scheme.w_qkv..],
        )
    }

    #[inline]
    pub fn w_o(&self) -> Tensor<&[DevByte]> {
        let d = self.scheme.nh * self.scheme.dh;
        let n = self.scheme.n as udim;
        Tensor::new(self.scheme.dt, &[d / n, d], &self.mem[self.scheme.w_o..])
    }

    #[inline]
    pub fn post_att_layernorm(&self) -> Tensor<&[DevByte]> {
        let d = self.scheme.nh * self.scheme.dh;
        Tensor::new(
            self.scheme.dt,
            &[d],
            &self.mem[self.scheme.post_att_layernorm..],
        )
    }

    #[inline]
    pub fn mlp_gate_up(&self) -> Tensor<&[DevByte]> {
        let di = self.scheme.di;
        let d = self.scheme.nh * self.scheme.dh;
        let n = self.scheme.n as udim;
        Tensor::new(
            self.scheme.dt,
            &[d, (di + di) / n],
            &self.mem[self.scheme.mlp_gate_up..],
        )
    }

    #[inline]
    pub fn mlp_down(&self) -> Tensor<&[DevByte]> {
        let di = self.scheme.di;
        let d = self.scheme.nh * self.scheme.dh;
        let n = self.scheme.n as udim;
        Tensor::new(
            self.scheme.dt,
            &[di / n, d],
            &self.mem[self.scheme.mlp_down..],
        )
    }
}

#[test]
fn test_load() {
    use common_nv::cuda::{self, Device};
    use log::LevelFilter::Trace;
    use simple_logger::SimpleLogger;

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    const N: usize = 1;

    cuda::init();
    if Device::count() < N {
        return;
    }

    SimpleLogger::new().with_level(Trace).init().unwrap();

    let time = Instant::now();
    let model = llama::Storage::load_safetensors(model_dir).unwrap();
    info!("mmap {:?}", time.elapsed());

    let contexts = (0..N as _)
        .map(|i| Device::new(i).retain_primary())
        .collect::<Vec<_>>();
    unsafe { ParameterMatrix::load(&model, &contexts).kill(&contexts) };
}
