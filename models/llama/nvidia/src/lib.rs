#![cfg(detected_cuda)]

mod resource;

#[macro_use]
extern crate log;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{upos, utok, FileLoadError};
use common_nv::{
    sample_nv, slice, udim, DataType, KernelRuntime, Kernels, NvidiaKernels, NvidiaKernelsPtx,
    Tensor,
};
use cuda::{
    ContextResource, ContextSpore, DevByte, DevMem, DevMemSpore, Device, EventSpore, HostMemSpore,
    Stream, StreamSpore,
};
use llama::{InferenceConfig, LayerStorage, Weight};
use resource::{DropOption, Resource};
use std::{
    cell::RefCell,
    collections::VecDeque,
    iter::repeat,
    ops::{Deref, DerefMut},
    path::Path,
    rc::Rc,
    slice::from_raw_parts,
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

pub use common_nv::{cuda, synchronize};
pub use resource::Cache;

pub struct Transformer {
    config: InferenceConfig,

    resource: Arc<Resource>,
    transfer: DropOption<StreamSpore>,
    kernels: DropOption<NvidiaKernels>,

    embed_tokens: Tensor<DropOption<HostMemSpore>>,
    layers: Vec<LayerStorage<HostMemSpore>>,
    lm_layernorm: Tensor<DropOption<DevMemSpore>>,
    lm_head: Tensor<DropOption<DevMemSpore>>,

    pool: Mutex<VecDeque<(LayerStorage<DevMemSpore>, EventSpore)>>,
}

pub struct ModelLoadMeta {
    pub device: Device,
    pub load_layers: usize,
}

impl ModelLoadMeta {
    #[inline]
    pub fn load_all_to(n: i32) -> Self {
        Self {
            device: Device::new(n),
            load_layers: usize::MAX,
        }
    }
}

impl Model for Transformer {
    type Meta = ModelLoadMeta;
    type Error = FileLoadError;

    #[inline]
    fn load(
        model_dir: impl AsRef<Path>,
        Self::Meta {
            device,
            load_layers,
        }: Self::Meta,
    ) -> Result<Self, Self::Error> {
        let time = Instant::now();
        let host = llama::Storage::load_safetensors(model_dir)?;
        info!("load host: {:?}", time.elapsed());
        let load_layers = (load_layers as udim).min(host.config.nlayers);

        device.set_mempool_threshold(u64::MAX);
        let resource = Arc::new(Resource::new(&device));
        resource.apply(|compute| {
            let ctx = compute.ctx();
            let transfer = ctx.stream();

            let page_lock = |u: &Weight| {
                let mut host = ctx.malloc_host::<u8>(u.len());
                host.copy_from_slice(u);
                host.sporulate()
            };
            let from_host = |u: &HostMemSpore| transfer.from_host(u).sporulate();

            let layers = host
                .layers
                .iter()
                .map(|l| l.map(page_lock))
                .collect::<Vec<_>>();
            let pool = layers
                .iter()
                .take(load_layers as usize)
                .map(|l| (l.map(from_host), transfer.record().sporulate()))
                .collect();

            Ok(Self {
                kernels: NvidiaKernelsPtx::new(
                    &[device],
                    host.config.d as _,
                    host.config.max_seq_len as _,
                )
                .load(&transfer)
                .into(),
                embed_tokens: host
                    .embed_tokens
                    .as_ref()
                    .map_physical(page_lock)
                    .map_physical(|u| u.into()),
                layers,
                lm_layernorm: host
                    .lm_layernorm
                    .map_physical(|u| transfer.from_host(&u).sporulate().into()),
                lm_head: host
                    .lm_head
                    .map_physical(|u| transfer.from_host(&u).sporulate().into()),
                pool: Mutex::new(pool),

                config: host.config,
                resource: resource.clone(),
                transfer: transfer.sporulate().into(),
            })
        })
    }
}

impl Transformer {
    #[inline]
    fn cache(&self, len: usize) -> Cache {
        Cache::new(&self.resource, len)
    }

    #[inline]
    fn tensor(&self, shape: &[udim]) -> Tensor<Cache> {
        Tensor::alloc(self.config.dt, shape, |len| self.cache(len))
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.config.max_seq_len
    }
    #[inline]
    fn eos_token(&self) -> utok {
        self.config.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.config.new_cache(|len| self.cache(len))
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        self.config.duplicate_cache(
            cache,
            pos,
            |len| self.cache(len),
            |dst, src| {
                self.resource.apply(|stream| {
                    let ctx = stream.ctx();
                    self.kernels.as_ref().on(stream).reform(
                        &mut dst.map_physical(|u| &mut **u.mem.as_mut().sprout_mut(ctx)),
                        &src.map_physical(|u| &**u.mem.as_ref().sprout_ref(ctx)),
                    );
                })
            },
        )
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;
        let d = self.config.d;

        let mut x = self.tensor(&[nt, d]);
        self.resource.apply(|compute| {
            self.kernels.as_ref().on(compute).gather(
                &mut x
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.as_mut().sprout_mut(compute.ctx())),
                &self.embed_tokens.as_ref().map_physical(|u| &**u.as_ref()),
                tokens,
            )
        });
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        self.resource.apply(|compute| {
            let ctx = compute.ctx();
            let transfer = self.transfer.as_ref().sprout_ref(ctx);
            let stream = ComputeStream {
                nh: self.config.nh,
                nkvh: self.config.nkvh,
                di: self.config.di,
                epsilon: self.config.epsilon,
                theta: self.config.theta,
                kernels: self.kernels.as_ref().on(compute),
                compute,
                transfer,
                host: &self.layers,
                dev: Rc::new(RefCell::new(self.pool.lock().unwrap())),
            };
            <ComputeStream as llama::ComputeStream>::forward(&stream, queries, token_embedded)
        })
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        self.resource.apply(|compute| {
            let ctx = compute.ctx();
            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| &mut **u.mem.as_mut().sprout_mut(ctx));
            let range =
                DecodingMeta::select(&mut x, decoding, |dst, src| compute.memcpy_d2d(dst, src));
            if range.is_empty() {
                return self.tensor(&[0, self.config.d]);
            }

            let lm_layernorm = self
                .lm_layernorm
                .as_ref()
                .map_physical(|u| &**u.as_ref().sprout_ref(ctx));
            let lm_head = self
                .lm_head
                .as_ref()
                .map_physical(|u| &**u.as_ref().sprout_ref(ctx));

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = self.tensor(&[x.shape()[0], lm_head.shape()[1]]);

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            let kernels = self.kernels.as_ref().on(compute);
            kernels.rms_norm(&mut x, &x_, &lm_layernorm, self.config.epsilon);
            kernels.mat_mul(
                &mut logits
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.as_mut().sprout_mut(ctx)),
                0.,
                &x,
                &lm_head,
                1.,
            );

            logits
        })
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        assert_eq!(logits.data_type(), DataType::F16);
        let &[_nt, voc] = logits.shape() else {
            panic!()
        };
        let voc = voc as usize;

        self.resource.apply(|compute| {
            sample_nv(
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
                    .enumerate(),
                logits
                    .take_physical()
                    .mem
                    .as_ref()
                    .sprout_ref(compute.ctx()),
                voc,
                compute,
            )
        })
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        self.resource.apply(|compute| {
            let ctx = compute.ctx();
            self.transfer.take().sprout(ctx);
            self.kernels.take().kill(ctx);
            self.embed_tokens.physical_mut().take().sprout(ctx);
            self.lm_layernorm.physical_mut().take().sprout(ctx);
            self.lm_head.physical_mut().take().sprout(ctx);
            while let Some(layer) = self.layers.pop() {
                layer.att_layernorm.take_physical().sprout(ctx);
                layer.att_qkv.take_physical().sprout(ctx);
                layer.att_o.take_physical().sprout(ctx);
                layer.mlp_layernorm.take_physical().sprout(ctx);
                layer.mlp_gate_up.take_physical().sprout(ctx);
                layer.mlp_down.take_physical().sprout(ctx);
            }
            let mut pool = self.pool.lock().unwrap();
            while let Some((layer, event)) = pool.pop_front() {
                layer.att_layernorm.take_physical().sprout(ctx);
                layer.att_qkv.take_physical().sprout(ctx);
                layer.att_o.take_physical().sprout(ctx);
                layer.mlp_layernorm.take_physical().sprout(ctx);
                layer.mlp_gate_up.take_physical().sprout(ctx);
                layer.mlp_down.take_physical().sprout(ctx);
                event.sprout(ctx);
            }
        });
    }
}

struct ComputeStream<'a> {
    nh: udim,
    nkvh: udim,
    di: udim,
    epsilon: f32,
    theta: f32,
    kernels: KernelRuntime<'a>,
    compute: &'a Stream<'a>,
    transfer: &'a Stream<'a>,
    host: &'a [LayerStorage<HostMemSpore>],
    dev: DevMemPool<'a>,
}

type DevMemPool<'a> =
    Rc<RefCell<MutexGuard<'a, VecDeque<(LayerStorage<DevMemSpore>, EventSpore)>>>>;

impl<'a> llama::ComputeStream for ComputeStream<'a> {
    type Byte = DevByte;
    type Storage = Cache;
    type Buf<'m> = DevMem<'m>;
    type Pos<'m> = DevMem<'m>;

    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        self.compute.malloc::<u8>(len)
    }
    fn free(&self, mem: Self::Buf<'_>) {
        mem.drop_on(self.compute);
    }
    fn map_pos<'b>(&self, pos: &'b [u32]) -> Self::Pos<'b>
    where
        Self: 'b,
    {
        self.compute.from_host(pos)
    }
    fn free_pos(&self, mem: Self::Pos<'_>) {
        mem.drop_on(self.compute);
    }
    fn map_storage<'b>(&'b self, storage: &'b mut Self::Storage) -> &'b mut [Self::Byte] {
        storage.mem.as_mut().sprout_mut(self.compute.ctx())
    }
    fn rms_norm<O, X, W>(&self, o: &mut Tensor<O>, x: &Tensor<X>, w: &Tensor<W>)
    where
        O: DerefMut<Target = [Self::Byte]>,
        X: Deref<Target = [Self::Byte]>,
        W: Deref<Target = [Self::Byte]>,
    {
        self.kernels.rms_norm(o, x, w, self.epsilon);
    }
    fn mat_mul<O, A, B>(
        &self,
        o: &mut Tensor<O>,
        beta: f32,
        a: &Tensor<A>,
        b: &Tensor<B>,
        alpha: f32,
    ) where
        O: DerefMut<Target = [Self::Byte]>,
        A: Deref<Target = [Self::Byte]>,
        B: Deref<Target = [Self::Byte]>,
    {
        self.kernels.mat_mul(o, beta, a, b, alpha);
    }
    fn rotary_embedding<X>(&self, x: &mut Tensor<X>, pos: &Tensor<Self::Pos<'_>>)
    where
        X: DerefMut<Target = [Self::Byte]>,
    {
        self.kernels.rotary_embedding(x, pos, self.theta);
    }
    fn reform<Y, X>(&self, y: &mut Tensor<Y>, x: &Tensor<X>)
    where
        Y: DerefMut<Target = [Self::Byte]>,
        X: Deref<Target = [Self::Byte]>,
    {
        self.kernels.reform(y, x);
    }
    fn softmax<X>(&self, x: &mut Tensor<X>)
    where
        X: DerefMut<Target = [Self::Byte]>,
    {
        self.kernels.softmax(x);
    }
    fn swiglu<A, B>(&self, a: &mut Tensor<A>, b: &Tensor<B>)
    where
        A: DerefMut<Target = [Self::Byte]>,
        B: Deref<Target = [Self::Byte]>,
    {
        self.kernels.swiglu(a, b);
    }
    fn nh(&self) -> udim {
        self.nh
    }
    fn nkvh(&self) -> udim {
        self.nkvh
    }
    fn di(&self) -> udim {
        self.di
    }
    fn layers(&self) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = Self::Byte>> {
        Iter::new(self.host, self.dev.clone(), self.compute, self.transfer)
    }
}

struct Iter<'a> {
    host: &'a [LayerStorage<HostMemSpore>],
    pool: DevMemPool<'a>,
    compute: &'a Stream<'a>,
    transfer: &'a Stream<'a>,
    layer: usize,
}

impl<'a> Iter<'a> {
    pub fn new(
        host: &'a [LayerStorage<HostMemSpore>],
        pool: DevMemPool<'a>,
        compute: &'a Stream,
        transfer: &'a Stream,
    ) -> Self {
        Self {
            host,
            pool,
            compute,
            transfer,
            layer: 0,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = LayerLoader<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.layer >= self.host.len() {
            return None;
        }

        let mut pool = self.pool.borrow_mut();
        let load = if pool.len() < self.host.len() {
            Some((self.layer + pool.len()) % self.host.len())
        } else {
            None
        };
        self.layer += 1;

        let (s, event) = pool.pop_front().unwrap();
        let ctx = self.compute.ctx();
        self.compute.wait_for(&event.sprout(ctx));

        Some(Self::Item {
            host: self.host,
            pool: self.pool.clone(),
            load,
            transfer: self.transfer,
            storage: Some(s),
        })
    }
}

struct LayerLoader<'a> {
    host: &'a [LayerStorage<HostMemSpore>],
    pool: DevMemPool<'a>,
    load: Option<usize>,
    transfer: &'a Stream<'a>,
    storage: Option<LayerStorage<DevMemSpore>>,
}

macro_rules! access {
    ($self:expr, $name:ident) => {
        $self
            .storage
            .as_ref()
            .unwrap()
            .$name
            .as_ref()
            .map_physical(|u| &**u.sprout_ref($self.transfer.ctx()))
    };
}
impl<'a> llama::LLamaLayer for LayerLoader<'a> {
    type Byte = DevByte;
    type Storage<'m> = &'m[DevByte] where Self: 'm;

    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_layernorm)
    }
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_qkv)
    }
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_o)
    }
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_layernorm)
    }
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_gate_up)
    }
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_down)
    }
}

impl Drop for LayerLoader<'_> {
    fn drop(&mut self) {
        let mut lll = self.storage.take().unwrap();
        if let Some(load) = self.load {
            macro_rules! exchange {
                ($($name:ident)+) => {
                    $(
                        let host = self.host[load].$name.physical();
                        let mut dev = lll.$name.physical_mut().sprout_mut(self.transfer.ctx());
                        self.transfer.memcpy_h2d(&mut dev, host);
                    )+
                };
            }
            exchange! {
                att_layernorm
                att_qkv
                att_o
                mlp_layernorm
                mlp_gate_up
                mlp_down
            }
        }
        self.pool
            .borrow_mut()
            .push_back((lll, self.transfer.record().sporulate()));
    }
}

#[test]
fn test_infer() {
    cuda::init();
    if let Some(device) = cuda::Device::fetch() {
        causal_lm::test_impl::<Transformer>(
            ModelLoadMeta {
                device,
                load_layers: 20,
            },
            &[
                29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592,
                21106, 29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
            ],
        );
    };
}
