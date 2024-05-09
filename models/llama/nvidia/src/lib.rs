#![cfg(detected_cuda)]

#[macro_use]
extern crate log;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{upos, utok, FileLoadError};
use common_nv::{
    cuda::{DevByte, DevMem, DevMemSpore, EventSpore, HostMemSpore, Stream},
    sample_nv, slice, udim, DataType, KernelRuntime, Kernels, NvidiaKernels, NvidiaKernelsPtx,
    Tensor,
};
use cuda::{Context, ContextResource, ContextSpore, Device, StreamSpore};
use llama::{InferenceConfig, LayerStorage, Weight};
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

pub struct Transformer {
    config: InferenceConfig,

    context: Arc<Context>,
    transfer: StreamSpore,
    kernels: NvidiaKernels,

    embed_tokens: Tensor<HostMemSpore>,
    layers: Vec<LayerStorage<HostMemSpore>>,
    lm_layernorm: Tensor<DevMemSpore>,
    lm_head: Tensor<DevMemSpore>,

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
        let context = Arc::new(device.retain_primary());
        context.apply(|ctx| {
            let transfer = ctx.stream();
            let block_size = ctx.dev().max_block_dims().0;

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
                context: context.clone(),

                kernels: NvidiaKernelsPtx::new(
                    host.config.d as _,
                    host.config.max_seq_len as _,
                    block_size,
                )
                .load(&transfer),
                embed_tokens: host.embed_tokens.as_ref().map_physical(page_lock),
                layers,
                lm_layernorm: host
                    .lm_layernorm
                    .map_physical(|u| transfer.from_host(&u).sporulate()),
                lm_head: host
                    .lm_head
                    .map_physical(|u| transfer.from_host(&u).sporulate()),
                pool: Mutex::new(pool),

                config: host.config,
                transfer: transfer.sporulate(),
            })
        })
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn eos_token(&self) -> utok {
        self.config.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.config.new_cache(|len| Cache {
            context: self.context.clone(),
            stream: None,
            mem: self.context.apply(|ctx| ctx.malloc::<u8>(len).sporulate()),
        })
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        self.context.apply(|ctx| {
            let stream = ctx.stream();
            self.config.duplicate_cache(
                cache,
                pos,
                |len| Cache {
                    context: self.context.clone(),
                    stream: None,
                    mem: stream.malloc::<u8>(len).sporulate(),
                },
                |dst, src| {
                    self.kernels.on(&stream).reform(
                        &mut dst.map_physical(|u| unsafe { u.mem.sprout(ctx) }),
                        &src.map_physical(|u| unsafe { u.mem.sprout(ctx) }),
                    );
                },
            )
        })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let dt = self.config.dt;
        let d = self.config.d;
        self.context.apply(|ctx| {
            let compute = ctx.stream();
            let kernels = self.kernels.on(&compute);

            let tokens = queries.into_iter().collect::<Vec<_>>();
            let nt = tokens.len() as udim;

            let mut x = Tensor::alloc(dt, &[nt, d], |len| compute.malloc::<u8>(len));
            kernels.gather(&mut x, &self.embed_tokens, tokens);
            x.map_physical(|u| Cache {
                context: self.context.clone(),
                stream: Some(compute.sporulate()),
                mem: u.sporulate(),
            })
        })
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        self.context.apply(|ctx| {
            let compute = unsafe {
                token_embedded
                    .physical()
                    .stream
                    .as_ref()
                    .unwrap()
                    .sprout(ctx)
            };
            let transfer = unsafe { self.transfer.sprout(ctx) };
            let stream = ComputeStream {
                nh: self.config.nh,
                nkvh: self.config.nkvh,
                di: self.config.di,
                epsilon: self.config.epsilon,
                theta: self.config.theta,
                kernels: self.kernels.on(&compute),
                compute: &compute,
                transfer: &transfer,
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
        let dt = self.config.dt;
        let d = self.config.d;

        self.context.apply(|ctx| {
            let stream_spore = hidden_state.physical_mut().stream.take().unwrap();
            let stream = unsafe { stream_spore.sprout(ctx) };

            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| unsafe { u.mem.sprout(ctx) });
            let range =
                DecodingMeta::select(&mut x, decoding, |dst, src| stream.memcpy_d2d(dst, src));
            if range.is_empty() {
                return Tensor::alloc(dt, &[0, d as _], |_| Cache {
                    context: self.context.clone(),
                    stream: Some(stream_spore),
                    mem: stream.malloc::<u8>(0).sporulate(),
                });
            }

            let lm_layernorm = self
                .lm_layernorm
                .as_ref()
                .map_physical(|u| unsafe { ctx.sprout(u) });
            let lm_head = self
                .lm_head
                .as_ref()
                .map_physical(|u| unsafe { ctx.sprout(u) });

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], |len| {
                stream.malloc::<u8>(len)
            });

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            let kernels = self.kernels.on(&stream);
            kernels.rms_norm(&mut x, &x_, &lm_layernorm, self.config.epsilon);
            kernels.mat_mul(&mut logits, 0., &x, &lm_head, 1.);

            logits.map_physical(|u| Cache {
                context: self.context.clone(),
                stream: Some(stream_spore),
                mem: u.sporulate(),
            })
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

        let Cache {
            context,
            stream,
            mem,
        } = logits.physical();
        context.apply(|ctx| {
            sample_nv(
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
                    .enumerate(),
                &unsafe { mem.sprout(ctx) },
                voc,
                &unsafe { stream.as_ref().unwrap().sprout(ctx) },
            )
        })
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        self.context.apply(|ctx| unsafe {
            ctx.kill(&mut self.transfer);
            ctx.kill(self.embed_tokens.physical_mut());
            ctx.kill(self.lm_layernorm.physical_mut());
            ctx.kill(self.lm_head.physical_mut());
            for layer in self.layers.iter_mut() {
                ctx.kill(layer.att_layernorm.physical_mut());
                ctx.kill(layer.att_qkv.physical_mut());
                ctx.kill(layer.att_o.physical_mut());
                ctx.kill(layer.mlp_layernorm.physical_mut());
                ctx.kill(layer.mlp_gate_up.physical_mut());
                ctx.kill(layer.mlp_down.physical_mut());
            }
            let mut pool = self.pool.lock().unwrap();
            while let Some((mut layer, mut event)) = pool.pop_front() {
                ctx.kill(layer.att_layernorm.physical_mut());
                ctx.kill(layer.att_qkv.physical_mut());
                ctx.kill(layer.att_o.physical_mut());
                ctx.kill(layer.mlp_layernorm.physical_mut());
                ctx.kill(layer.mlp_gate_up.physical_mut());
                ctx.kill(layer.mlp_down.physical_mut());
                ctx.kill(&mut event);
            }
            self.kernels.kill(ctx);
        });
    }
}

pub struct Cache {
    context: Arc<Context>,
    stream: Option<StreamSpore>,
    mem: DevMemSpore,
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.context.apply(|ctx| unsafe {
            if let Some(mut stream) = self.stream.take() {
                self.mem.kill_on(&ctx.sprout(&stream));
                ctx.kill(&mut stream);
            } else {
                ctx.kill(&mut self.mem);
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
    fn map_storage(&self, storage: &mut Self::Storage) -> impl DerefMut<Target = [Self::Byte]> {
        unsafe { storage.mem.sprout(self.compute.ctx()) }
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

        let (s, mut event) = pool.pop_front().unwrap();
        let ctx = self.compute.ctx();
        self.compute.wait_for(&unsafe { event.sprout(ctx) });
        unsafe { ctx.kill(&mut event) };

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
            .map_physical(|u| unsafe { u.sprout($self.transfer.ctx()) })
    };
}
impl<'a> llama::LLamaLayer for LayerLoader<'a> {
    type Byte = DevByte;
    type Storage<'m> = DevMem<'m> where Self: 'm;

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
        let lll = self.storage.take().unwrap();
        if let Some(load) = self.load {
            macro_rules! exchange {
                ($($name:ident)+) => {
                    $(
                        let host = self.host[load].$name.physical();
                        let mut dev = unsafe { lll.$name.physical().sprout(self.transfer.ctx()) };
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
