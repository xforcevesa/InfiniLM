//! Common types and functions used in transformer.

#![deny(warnings, missing_docs)]

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub type upos = u32;

mod between_f32;
mod blob;
pub mod safe_tensors;
pub mod test_model;

pub use between_f32::BetweenF32;
pub use blob::Blob;
pub use half::{bf16, f16};

/// 加载 safetensors 文件可能产生的错误。
#[derive(Debug)]
pub enum FileLoadError {
    /// IO 错误。
    Io(std::io::Error),
    /// Json 解析错误。
    Json(serde_json::Error),
}
