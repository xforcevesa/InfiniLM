use serde::{
    de::{Unexpected, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum DataType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl DataType {
    #[inline]
    pub const fn new<T: Ty>() -> Self {
        T::DATA_TYPE
    }

    #[inline]
    pub const fn size(&self) -> usize {
        match self {
            Self::Bool => <bool as Ty>::SIZE,
            Self::I8 => <i8 as Ty>::SIZE,
            Self::I16 => <i16 as Ty>::SIZE,
            Self::I32 => <i32 as Ty>::SIZE,
            Self::I64 => <i64 as Ty>::SIZE,
            Self::U8 => <u8 as Ty>::SIZE,
            Self::U16 => <u16 as Ty>::SIZE,
            Self::U32 => <u32 as Ty>::SIZE,
            Self::U64 => <u64 as Ty>::SIZE,
            Self::F16 => <half::f16 as Ty>::SIZE,
            Self::BF16 => <half::bf16 as Ty>::SIZE,
            Self::F32 => <f32 as Ty>::SIZE,
            Self::F64 => <f64 as Ty>::SIZE,
        }
    }
}

pub trait Ty: Sized {
    const SIZE: usize = std::mem::size_of::<Self>();
    const DATA_TYPE: DataType;
}

impl Ty for bool {
    const DATA_TYPE: DataType = DataType::Bool;
}
impl Ty for i8 {
    const DATA_TYPE: DataType = DataType::I8;
}
impl Ty for i16 {
    const DATA_TYPE: DataType = DataType::I16;
}
impl Ty for i32 {
    const DATA_TYPE: DataType = DataType::I32;
}
impl Ty for i64 {
    const DATA_TYPE: DataType = DataType::I64;
}
impl Ty for u8 {
    const DATA_TYPE: DataType = DataType::U8;
}
impl Ty for u16 {
    const DATA_TYPE: DataType = DataType::U16;
}
impl Ty for u32 {
    const DATA_TYPE: DataType = DataType::U32;
}
impl Ty for u64 {
    const DATA_TYPE: DataType = DataType::U64;
}
impl Ty for half::f16 {
    const DATA_TYPE: DataType = DataType::F16;
}
impl Ty for half::bf16 {
    const DATA_TYPE: DataType = DataType::BF16;
}
impl Ty for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}
impl Ty for f64 {
    const DATA_TYPE: DataType = DataType::F64;
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::F16 => serializer.serialize_str("float16"),
            Self::BF16 => serializer.serialize_str("bfloat16"),
            Self::F32 => serializer.serialize_str("float32"),
            _ => todo!(),
        }
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(DataTypeVisitor)
    }
}

struct DataTypeVisitor;

impl<'de> Visitor<'de> for DataTypeVisitor {
    type Value = DataType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "pytorch dtype string: \"float16\", \"bfloat16\", or \"float32\""
        )
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match v {
            "float16" => Ok(DataType::F16),
            "bfloat16" => Ok(DataType::BF16),
            "float32" => Ok(DataType::F32),
            _ => Err(E::invalid_value(
                Unexpected::Str(v),
                &"\"float16\", \"bfloat16\", or \"float32\"",
            )),
        }
    }
}
