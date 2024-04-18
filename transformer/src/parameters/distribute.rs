use crate::Llama2;
use common::Blob;
use std::sync::Arc;
use tensor::{slice, udim, DataType, Tensor};

pub struct DistributedLayer {
    scheme: Arc<DistributeScheme>,
    blob: Blob,
}

impl DistributedLayer {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.blob
    }

    #[inline]
    pub fn input_layernorm(&self) -> Tensor<&[u8]> {
        Tensor::new(
            self.scheme.dt,
            &[self.scheme.nh * self.scheme.dh],
            &self.blob[self.scheme.input_layernorm..],
        )
    }

    #[inline]
    pub fn w_qkv(&self) -> Tensor<&[u8]> {
        let nh = self.scheme.nh;
        let nkvh = self.scheme.nkvh;
        let dh = self.scheme.dh;
        let d = nh * dh;
        let n = self.scheme.n as udim;
        Tensor::new(
            self.scheme.dt,
            &[(nh + nkvh + nkvh) / n * dh, d],
            &self.blob[self.scheme.w_qkv..],
        )
    }

    #[inline]
    pub fn w_o(&self) -> Tensor<&[u8]> {
        let d = self.scheme.nh * self.scheme.dh;
        let n = self.scheme.n as udim;
        Tensor::new(self.scheme.dt, &[d / n, d], &self.blob[self.scheme.w_o..])
    }

    #[inline]
    pub fn post_att_layernorm(&self) -> Tensor<&[u8]> {
        Tensor::new(
            self.scheme.dt,
            &[self.scheme.nh * self.scheme.dh],
            &self.blob[self.scheme.post_att_layernorm..],
        )
    }

    #[inline]
    pub fn mlp_gate_up(&self) -> Tensor<&[u8]> {
        let di = self.scheme.di;
        let d = self.scheme.nh * self.scheme.dh;
        let n = self.scheme.n as udim;
        Tensor::new(
            self.scheme.dt,
            &[(di + di) / n, d],
            &self.blob[self.scheme.mlp_gate_up..],
        )
    }

    #[inline]
    pub fn mlp_down(&self) -> Tensor<&[u8]> {
        let di = self.scheme.di;
        let d = self.scheme.nh * self.scheme.dh;
        let n = self.scheme.n as udim;
        Tensor::new(
            self.scheme.dt,
            &[d, di / n],
            &self.blob[self.scheme.mlp_down..],
        )
    }
}

pub struct Distributer<'a> {
    model: &'a dyn Llama2,
    scheme: Arc<DistributeScheme>,
}

impl<'a> Distributer<'a> {
    #[inline]
    pub fn new(model: &'a dyn Llama2, n: usize, align: usize) -> Self {
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
        assert!(layer < self.model.num_hidden_layers());
        assert!(i < self.scheme.n);

        let dt = self.scheme.dt;
        let d = self.scheme.nh * self.scheme.dh;
        let nh = self.scheme.nh;
        let nkvh = self.scheme.nkvh;
        let dh = self.scheme.dh;
        let di = self.scheme.di;
        let n = self.scheme.n as udim;
        let i = i as udim;

        let mut blob = Blob::new(self.scheme.total_size);
        // layernorm
        self.model
            .input_layernorm(layer)
            .reform_to(&mut Tensor::new(
                dt,
                &[d],
                &mut blob[self.scheme.input_layernorm..],
            ));
        self.model
            .post_attention_layernorm(layer)
            .reform_to(&mut Tensor::new(
                dt,
                &[d],
                &mut blob[self.scheme.post_att_layernorm..],
            ));
        let shape_qkv = &[(nh + nkvh + nkvh) * dh / n, d];
        // wq
        {
            let w = nh / n * dh;
            self.model
                .self_attn_q_proj(layer)
                .slice(&[slice![from w*i, take w], slice![all]])
                .reform_to(
                    &mut Tensor::new(dt, shape_qkv, &mut blob[self.scheme.w_qkv..])
                        .slice(&[slice![take w], slice![all]]),
                );
        }
        // wk
        {
            let w = nkvh / n * dh;
            self.model
                .self_attn_k_proj(layer)
                .slice(&[slice![from w*i, take w], slice![all]])
                .reform_to(
                    &mut Tensor::new(dt, shape_qkv, &mut blob[self.scheme.w_qkv..])
                        .slice(&[slice![from d/n, take w], slice![all]]),
                );
        }
        // wv
        {
            let w = nkvh / n * dh;
            self.model
                .self_attn_v_proj(layer)
                .slice(&[slice![from w*i, take w], slice![all]])
                .reform_to(
                    &mut Tensor::new(dt, shape_qkv, &mut blob[self.scheme.w_qkv..])
                        .slice(&[slice![from d/n + w, take w], slice![all]]),
                );
        }
        // wo
        {
            let w = nh / n * dh;
            self.model
                .self_attn_o_proj(layer)
                .slice(&[slice![all], slice![from w*i, take w]])
                .reform_to(&mut Tensor::new(dt, &[d, w], &mut blob[self.scheme.w_o..]));
        }
        let shape_gate_up = &[(di + di) / n, d];
        // mlp gate
        {
            let w = di / n;
            self.model
                .mlp_gate(layer)
                .slice(&[slice![from w*i, take w], slice![all]])
                .reform_to(
                    &mut Tensor::new(dt, shape_gate_up, &mut blob[self.scheme.mlp_gate_up..])
                        .slice(&[slice![take w], slice![all]]),
                );
        }
        // mlp up
        {
            let w = di / n;
            self.model
                .mlp_up(layer)
                .slice(&[slice![from w*i, take w], slice![all]])
                .reform_to(
                    &mut Tensor::new(dt, shape_gate_up, &mut blob[self.scheme.mlp_gate_up..])
                        .slice(&[slice![from w, take w], slice![all]]),
                );
        }
        // mlp down
        {
            let w = di / n;
            self.model
                .mlp_down(layer)
                .slice(&[slice![all], slice![from w*i, take w]])
                .reform_to(&mut Tensor::new(
                    dt,
                    &[d, w],
                    &mut blob[self.scheme.mlp_down..],
                ));
        }

        DistributedLayer {
            scheme: self.scheme.clone(),
            blob,
        }
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
    fn new(model: &dyn Llama2, n: usize, align: usize) -> Arc<Self> {
        assert_eq!(model.num_key_value_heads() % n, 0);
        assert_eq!(model.intermediate_size() % n, 0);
        assert!(align.is_power_of_two());

        let d = model.hidden_size();
        let nh = model.num_attention_heads();
        let nkvh = model.num_key_value_heads();
        let dh = d / nh;
        let di = model.intermediate_size();

        let mut size = 0usize;
        let mut align = |addition: usize| -> usize {
            let offset = (size + (align - 1)) & !(align - 1);
            size = offset + addition * model.data_type().size();
            offset
        };

        let offset_input_layernorm = align(d);
        let offset_w_qkv = align((nh + nkvh + nkvh) * dh * d / n);
        let offset_w_o = align(d * d / n);
        let offset_post_att_layernorm = align(d);
        let offset_mlp_gate_up = align((di + di) * d / n);
        let offset_mlp_down = align(di * d / n);
        let total_size = size;

        Arc::new(Self {
            dt: model.data_type(),
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

#[test]
fn test() {
    use super::Memory;
    use common::safe_tensors::SafeTensorsError;
    use std::{io::ErrorKind::NotFound, time::Instant};

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let time = Instant::now();
    let safetensors = Memory::load_safetensors(model_dir);
    println!("mmap {:?}", time.elapsed());

    let model = match safetensors {
        Ok(m) => m,
        Err(SafeTensorsError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };

    let distributer = Distributer::new(&model, 4, 512);
    let time = Instant::now();
    for layer in 0..model.num_hidden_layers() {
        for i in 0..4 {
            let _ = distributer.distribute(layer, i);
        }
    }
    println!("distribute {:?}", time.elapsed());
}
