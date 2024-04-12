//! Common types and functions used in transformer.

#![deny(warnings, missing_docs)]

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub type upos = u32;

mod blob;
pub mod test_model;

pub use blob::Blob;
