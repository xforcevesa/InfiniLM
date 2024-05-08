use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{f16, upos, utok, Blob, FileLoadError};
use common_cpu::{gather, mat_mul, rms_norm, rotary_embedding, softmax, swiglu};
use llama::{ComputeStream, LayerStorage, Storage, Weight};
use std::{
    iter::repeat,
    ops::{Deref, DerefMut},
    path::Path,
    slice::from_raw_parts,
};
use tensor::{reslice, slice, udim, Tensor};

pub struct Transformer(Storage);

impl Model for Transformer {
    type Meta = ();
    type Error = FileLoadError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, _meta: Self::Meta) -> Result<Self, Self::Error> {
        Ok(Self(llama::Storage::load_safetensors(model_dir)?))
    }
}

impl ComputeStream for Transformer {
    type Byte = u8;
    type Storage = Blob;
    type Buf<'m> = Blob;
    type Pos<'m> = &'m [u8];

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        Blob::new(len)
    }
    #[inline]
    fn map_pos<'p>(&self, pos: &'p [u32]) -> Self::Pos<'p>
    where
        Self: 'p,
    {
        reslice(pos)
    }
    fn map_storage(&self, storage: &mut Self::Storage) -> impl DerefMut<Target = [Self::Byte]> {
        &mut **storage
    }
    #[inline]
    fn rms_norm<O, X, W>(&self, o: &mut Tensor<O>, x: &Tensor<X>, w: &Tensor<W>)
    where
        O: DerefMut<Target = [Self::Byte]>,
        X: Deref<Target = [Self::Byte]>,
        W: Deref<Target = [Self::Byte]>,
    {
        rms_norm(o, x, w, self.0.config.epsilon);
    }
    #[inline]
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
        mat_mul(o, beta, a, b, alpha);
    }
    #[inline]
    fn rotary_embedding<X>(&self, x: &mut Tensor<X>, pos: &Tensor<Self::Pos<'_>>)
    where
        X: DerefMut<Target = [Self::Byte]>,
    {
        rotary_embedding(x, pos, self.0.config.theta);
    }
    #[inline]
    fn reform<Y, X>(&self, y: &mut Tensor<Y>, x: &Tensor<X>)
    where
        Y: DerefMut<Target = [Self::Byte]>,
        X: Deref<Target = [Self::Byte]>,
    {
        x.reform_to(y);
    }
    #[inline]
    fn softmax<X>(&self, x: &mut Tensor<X>)
    where
        X: DerefMut<Target = [Self::Byte]>,
    {
        softmax(x);
    }
    #[inline]
    fn swiglu<A, B>(&self, a: &mut Tensor<A>, b: &Tensor<B>)
    where
        A: DerefMut<Target = [Self::Byte]>,
        B: Deref<Target = [Self::Byte]>,
    {
        swiglu(a, b);
    }
    #[inline]
    fn nh(&self) -> udim {
        self.0.config.nh
    }
    #[inline]
    fn nkvh(&self) -> udim {
        self.0.config.nkvh
    }
    #[inline]
    fn di(&self) -> udim {
        self.0.config.di
    }
    #[inline]
    fn layers(&self) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = Self::Byte>> {
        self.0.layers.iter().map(LlamaLayer)
    }
}

struct LlamaLayer<'a>(&'a LayerStorage<Weight>);

impl<'a> llama::LLamaLayer for LlamaLayer<'a> {
    type Byte = u8;
    type Storage<'m> = Weight where Self: 'm;

    #[inline]
    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        self.0.att_layernorm.clone()
    }
    #[inline]
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        self.0.att_qkv.clone()
    }
    #[inline]
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        self.0.att_o.clone()
    }
    #[inline]
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        self.0.mlp_layernorm.clone()
    }
    #[inline]
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        self.0.mlp_gate_up.clone()
    }
    #[inline]
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        self.0.mlp_down.clone()
    }
}

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn eos_token(&self) -> utok {
        self.0.config.eos_token
    }
    #[inline]
    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.0.config.new_cache(Blob::new)
    }
    #[inline]
    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        self.0
            .config
            .duplicate_cache(cache, pos, Blob::new, |dst, src| {
                src.map_physical(|u| &**u)
                    .reform_to(&mut dst.map_physical(|u| &mut **u))
            })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let dt = self.0.config.dt;
        let d = self.0.config.d;

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = Tensor::alloc(dt, &[nt, d], Blob::new);
        gather(&mut x, &self.0.embed_tokens, tokens);
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        <Self as ComputeStream>::forward(self, queries, token_embedded)
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.0.config.dt;
        let d = self.0.config.d;

        let mut x = hidden_state;
        let range = DecodingMeta::select(&mut x, decoding, |dst, src| dst.copy_from_slice(src));

        if range.is_empty() {
            return Tensor::alloc(dt, &[0, d as _], Blob::new);
        }

        let lm_layernorm = &self.0.lm_layernorm;
        let lm_head = &self.0.lm_head;
        let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
        let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        rms_norm(&mut x, &x_, lm_layernorm, self.0.config.epsilon);
        mat_mul(&mut logits, 0., &x, lm_head, 1.);

        logits
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let &[_, voc] = logits.shape() else { panic!() };
        let logits: &[f16] = reslice(logits.as_slice());
        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| args.random(&common_cpu::slice!(logits; voc; [i])))
            .collect()
    }
}

#[test]
fn test_infer() {
    causal_lm::test_impl::<Transformer>(
        (),
        &[
            29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592, 21106,
            29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
        ],
    );
}
