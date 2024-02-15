use model_parameters::{DataType, Llama2, Memory};

pub struct Transformer {
    model: Box<dyn Llama2>,
}

impl<T> From<T> for Transformer
where
    T: 'static + Llama2,
{
    fn from(value: T) -> Self {
        let model: Box<dyn Llama2> = if value.data_type() == DataType::BF16 {
            Box::new(Memory::cast(&value, DataType::F32))
        } else {
            Box::new(value)
        };
        Self { model }
    }
}
