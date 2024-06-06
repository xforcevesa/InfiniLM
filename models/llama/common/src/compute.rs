use causal_lm::QueryContext;
use itertools::izip;
use std::ops::{Deref, DerefMut};
use tensor::{slice, split, udim, LocalSplitable, Tensor};

pub trait ComputeStream {
    type Byte;
    type Storage;
    type Buf<'m>: DerefMut<Target = [Self::Byte]>;
    type Pos<'m>: Deref<Target = [Self::Byte]>;

    fn malloc(&self, len: usize) -> Self::Buf<'_>;
    fn free(&self, _mem: Self::Buf<'_>) {}
    fn map_pos<'p>(&self, pos: &'p [u32]) -> Self::Pos<'p>
    where
        Self: 'p;
    fn free_pos(&self, _mem: Self::Pos<'_>) {}
    fn map_storage<'a>(&'a self, storage: &'a mut Self::Storage) -> &'a mut [Self::Byte];

    fn rms_norm<O, X, W>(&self, o: &mut Tensor<O>, x: &Tensor<X>, w: &Tensor<W>)
    where
        O: DerefMut<Target = [Self::Byte]>,
        X: Deref<Target = [Self::Byte]>,
        W: Deref<Target = [Self::Byte]>;

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
        B: Deref<Target = [Self::Byte]>;

    fn rotary_embedding<X>(&self, x: &mut Tensor<X>, pos: &Tensor<Self::Pos<'_>>)
    where
        X: DerefMut<Target = [Self::Byte]>;

    fn reform<Y, X>(&self, y: &mut Tensor<Y>, x: &Tensor<X>)
    where
        Y: DerefMut<Target = [Self::Byte]>,
        X: Deref<Target = [Self::Byte]>;

    fn softmax<X>(&self, x: &mut Tensor<X>)
    where
        X: DerefMut<Target = [Self::Byte]>;

    fn swiglu<A, B>(&self, a: &mut Tensor<A>, b: &Tensor<B>)
    where
        A: DerefMut<Target = [Self::Byte]>,
        B: Deref<Target = [Self::Byte]>;

    fn nh(&self) -> udim;
    fn nkvh(&self) -> udim;
    fn di(&self) -> udim;
    fn layers(&self) -> impl Iterator<Item = impl LLamaLayer<Byte = Self::Byte>>;

    fn forward<'q>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'q, Self::Storage>>,
        mut token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self::Storage: 'q,
    {
        let mut queries = queries.into_iter().collect::<Vec<_>>();
        let mut nt = 0;
        let mut max_seq_len = 0;
        let mut max_att_len = 0;
        let seq_len = queries
            .iter()
            .map(|q| {
                let seq = q.seq_len();
                let att = q.att_len();
                nt += seq;
                max_seq_len = max_seq_len.max(seq);
                max_att_len = max_att_len.max(att);
                seq
            })
            .collect::<Vec<_>>();

        let dt = token_embedded.data_type();
        let d = token_embedded.shape()[1];
        let nh = self.nh();
        let nkvh = self.nkvh();
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.di();
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();

        let mut x = token_embedded
            .as_mut()
            .map_physical(|u| self.map_storage(u));
        let reusing = (d + dkv + dkv).max(di + di);
        let mut state_buf = Tensor::alloc(dt, &[nt, d + reusing], |len| self.malloc(len));

        let mut q_buf = self.malloc((nh * max_seq_len * dh) as usize * dt.size());
        let mut att_buf = self.malloc((nh * max_seq_len * max_att_len) as usize * dt.size());
        let pos = causal_lm::pos(&queries, nt);
        let pos = pos.as_ref().map_physical(|u| self.map_pos(u));

        for (layer, params) in self.layers().enumerate() {
            let (mut x1, qkv) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing);
            let mut qkv = qkv.slice(&[slice![=>], slice![=> d + dkv + dkv]]);

            self.rms_norm(&mut x1, &x, &params.att_layernorm());
            self.mat_mul(&mut qkv, 0., &x1, &params.att_qkv(), 1.);

            let (q, k, v) = split!(qkv; [1]: d, dkv, dkv);
            let mut q = q.reshape(&[nt, nh, dh]);
            let mut k = k.reshape(&[nt, nkvh, dh]);
            let v = v.reshape(&[nt, nkvh, dh]);
            let o = x1.reshape(&[nt, nh, dh]);

            self.rotary_embedding(&mut q, &pos);
            self.rotary_embedding(&mut k, &pos);

            let q = q.transpose(&[1, 0, 2]).split(1, &seq_len);
            let k = k.transpose(&[1, 0, 2]).split(1, &seq_len);
            let v = v.transpose(&[1, 0, 2]).split(1, &seq_len);
            let o = o.transpose(&[1, 0, 2]).split(1, &seq_len);

            for (query, q, k, v, mut o) in izip!(&mut queries, q, k, v, o) {
                let pos = query.pos();
                let seq_len = query.seq_len();
                let att_len = query.att_len();
                let mut cache = query
                    .cache
                    .as_mut()
                    .map(|t| t.as_mut().map_physical(|u| self.map_storage(u)));
                let mut query = QueryContext {
                    cache: cache.as_mut(),
                    range: query.range.clone(),
                };
                let Some((mut k_cache, mut v_cache)) = query.cache(layer as _) else {
                    continue;
                };

                let slice_cat = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
                let slice_att = &[slice![=>], slice![      => att_len], slice![=>]];
                let shape_q0 = &[nkvh * head_group, seq_len, dh];
                let shape_q1 = &[nkvh, head_group * seq_len, dh];
                let shape_att0 = &[nkvh, head_group * seq_len, att_len];
                let shape_att1 = &[nkvh * head_group, seq_len, att_len];

                let mut q_att = Tensor::new(dt, shape_q0, &mut q_buf[..]);
                let mut k_cat = k_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                let mut v_cat = v_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                self.reform(&mut q_att, &q);
                self.reform(&mut k_cat, &k);
                self.reform(&mut v_cat, &v);

                let q_att = q_att.reshape(shape_q1);
                let k_att = k_cache.slice(slice_att).transpose(&[0, 2, 1]);
                let v_att = v_cache.slice(slice_att);

                let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
                self.mat_mul(&mut att, 0., &q_att, &k_att, head_div);
                let mut att = att.reshape(shape_att1);
                self.softmax(&mut att);
                let mut x2 = q_att;
                self.mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1.);

                self.reform(&mut o, &x2.reshape(shape_q0));
            }

            let (mut x1, gate_up) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing);
            let mut gate_up = gate_up.slice(&[slice![=>], slice![=> di + di]]);

            self.mat_mul(&mut x, 1., &x1, &params.att_o(), 1.);
            self.rms_norm(&mut x1, &x, &params.mlp_layernorm());
            self.mat_mul(&mut gate_up, 0., &x1, &params.mlp_gate_up(), 1.);
            let (mut gate, up) = split!(gate_up; [1]: di, di);
            self.swiglu(&mut gate, &up);
            self.mat_mul(&mut x, 1., &gate, &params.mlp_down(), 1.);
        }
        self.free_pos(pos.take_physical());
        self.free(state_buf.take_physical());
        self.free(q_buf);
        self.free(att_buf);
        drop(x);
        token_embedded
    }
}

pub trait LLamaLayer {
    type Byte;
    type Storage<'m>: Deref<Target = [Self::Byte]>
    where
        Self: 'm;

    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>>;
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>>;
    fn att_o(&self) -> Tensor<Self::Storage<'_>>;
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>>;
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>>;
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>>;
}
