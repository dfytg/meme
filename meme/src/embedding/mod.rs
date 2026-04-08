//! Embedding model abstraction — unified interface for API and local ONNX providers.

mod api;
mod embedder;
#[cfg(feature = "onnx")]
mod onnx;

pub(crate) use api::ApiEmbedding;
pub(crate) use embedder::Embedder;
#[cfg(feature = "onnx")]
pub(crate) use onnx::OnnxEmbedding;
