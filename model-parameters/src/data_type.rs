use serde::{
    de::{Unexpected, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum DataType {
    F16,
    BF16,
    F32,
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
        }
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

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(DataTypeVisitor)
    }
}

impl DataType {
    #[inline]
    pub const fn size(&self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}
