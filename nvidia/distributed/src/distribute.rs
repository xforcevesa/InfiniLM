use common::Blob;
use common_nv::{slice, split, udim, DataType, Tensor};
use std::{ops::Deref, sync::Arc};

pub struct DistributedLayer(Blob);

impl Deref for DistributedLayer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Distributer<'a> {
    model: &'a llama::Storage,
    scheme: Arc<DistributeScheme>,
}

impl<'a> Distributer<'a> {
    #[inline]
    pub fn new(model: &'a llama::Storage, n: usize, align: usize) -> Self {
        Self {
            model,
            scheme: DistributeScheme::new(model, n, align),
        }
    }

    #[inline]
    pub fn scheme(&self) -> &DistributeScheme {
        &self.scheme
    }

    pub fn distribute(&self, layer: usize, i: usize) -> DistributedLayer {
        let &DistributeScheme {
            dt,
            nh,
            nkvh,
            dh,
            di,
            n,
            input_layernorm,
            w_qkv,
            w_o,
            post_att_layernorm,
            mlp_gate_up,
            mlp_down,
            total_size,
        } = &*self.scheme;

        assert!(layer < self.model.config.nlayers as usize);
        assert!(i < n);

        let d = nh * dh;
        let dkv = nkvh * dh;
        let n = self.scheme.n as udim;
        let i = i as udim;

        let mut blob = Blob::new(total_size);
        let layer = &self.model.layers[layer];
        // layernorm
        layer
            .att_layernorm
            .reform_to(&mut Tensor::new(dt, &[d], &mut blob[input_layernorm..]));
        layer
            .mlp_layernorm
            .reform_to(&mut Tensor::new(dt, &[d], &mut blob[post_att_layernorm..]));
        // wo
        {
            let w = nh / n * dh;
            layer
                .att_o
                .as_ref()
                .slice(&[slice![w*i =>=> w], slice![=>]])
                .map_physical(|u| &**u)
                .reform_to(&mut Tensor::new(dt, &[w, d], &mut blob[w_o..]));
        }
        // mlp down
        {
            let w = di / n;
            layer
                .mlp_down
                .as_ref()
                .slice(&[slice![w*i =>=> w], slice![=>]])
                .map_physical(|u| &**u)
                .reform_to(&mut Tensor::new(dt, &[w, d], &mut blob[mlp_down..]));
        }
        let shape_qkv = &[d, (nh + nkvh + nkvh) * dh / n];
        let (q, k, v) = split!(layer.att_qkv; [1]: d, dkv, dkv);
        // wq
        {
            let w = nh / n * dh;
            q.slice(&[slice![=>], slice![w*i =>=> w]]).reform_to(
                &mut Tensor::new(dt, shape_qkv, &mut blob[w_qkv..])
                    .slice(&[slice![=>], slice![=> w]]),
            );
        }
        // wk
        {
            let w = nkvh / n * dh;
            k.slice(&[slice![=>], slice![w*i =>=> w]]).reform_to(
                &mut Tensor::new(dt, shape_qkv, &mut blob[w_qkv..])
                    .slice(&[slice![=>], slice![d/n =>=> w]]),
            );
        }
        // wv
        {
            let w = nkvh / n * dh;
            v.slice(&[slice![=>], slice![w*i =>=> w]]).reform_to(
                &mut Tensor::new(dt, shape_qkv, &mut blob[w_qkv..])
                    .slice(&[slice![=>], slice![d/n + w =>=> w]]),
            );
        }
        let shape_gate_up = &[d, (di + di) / n];
        let (gate, up) = split!(layer.mlp_gate_up; [1]: di, di);
        // mlp gate
        {
            let w = di / n;
            gate.slice(&[slice![=>], slice![w*i =>=> w]]).reform_to(
                &mut Tensor::new(dt, shape_gate_up, &mut blob[mlp_gate_up..])
                    .slice(&[slice![=>], slice![=> w]]),
            );
        }
        // mlp up
        {
            let w = di / n;
            up.slice(&[slice![=>], slice![w*i =>=> w]]).reform_to(
                &mut Tensor::new(dt, shape_gate_up, &mut blob[mlp_gate_up..])
                    .slice(&[slice![=>], slice![w =>=> w]]),
            );
        }

        DistributedLayer(blob)
    }
}

#[derive(Clone, Debug)]
pub struct DistributeScheme {
    /// data type
    pub dt: DataType,
    /// num heads
    pub nh: udim,
    /// num kv heads
    pub nkvh: udim,
    /// head dim
    pub dh: udim,
    /// intermediate size
    pub di: udim,

    pub n: usize,
    pub input_layernorm: usize,
    pub w_qkv: usize,
    pub w_o: usize,
    pub post_att_layernorm: usize,
    pub mlp_gate_up: usize,
    pub mlp_down: usize,
    pub total_size: usize,
}

impl DistributeScheme {
    #[inline]
    fn new(model: &llama::Storage, n: usize, align: usize) -> Arc<Self> {
        assert_eq!(model.config.nkvh as usize % n, 0);
        assert_eq!(model.config.di as usize % n, 0);
        assert!(align.is_power_of_two());

        let d = model.config.d as usize;
        let nh = model.config.nh as usize;
        let nkvh = model.config.nkvh as usize;
        let dh = d / nh;
        let di = model.config.di as usize;

        let mut size = 0usize;
        let mut align = |addition: usize| -> usize {
            let offset = (size + (align - 1)) & !(align - 1);
            size = offset + addition * model.config.dt.size();
            offset
        };

        let offset_input_layernorm = align(d);
        let offset_w_qkv = align(d * (nh + nkvh + nkvh) * dh / n);
        let offset_w_o = align(d / n * d);
        let offset_post_att_layernorm = align(d);
        let offset_mlp_gate_up = align(d * (di + di) / n);
        let offset_mlp_down = align(di / n * d);
        let total_size = size;

        Arc::new(Self {
            dt: model.config.dt,
            nh: nh as _,
            nkvh: nkvh as _,
            dh: dh as _,
            di: di as _,

            n,
            input_layernorm: offset_input_layernorm,
            w_qkv: offset_w_qkv,
            w_o: offset_w_o,
            post_att_layernorm: offset_post_att_layernorm,
            mlp_gate_up: offset_mlp_gate_up,
            mlp_down: offset_mlp_down,
            total_size,
        })
    }
}

// #[test]
// fn test() {
//     use super::Memory;
//     use std::time::Instant;

//     let Some(model_dir) = common::test_model::find() else {
//         return;
//     };
//     println!("model_dir: {}", model_dir.display());

//     let time = Instant::now();
//     let model = Memory::load_safetensors(model_dir).unwrap();
//     println!("mmap {:?}", time.elapsed());

//     let distributer = Distributer::new(&model, 4, 512);
//     let time = Instant::now();
//     for layer in 0..model.num_hidden_layers() {
//         for i in 0..4 {
//             let _ = distributer.distribute(layer, i);
//         }
//     }
//     println!("distribute {:?}", time.elapsed());
// }
