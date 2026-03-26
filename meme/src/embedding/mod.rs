//! Embedding model abstraction — unified interface for API and local ONNX providers.

mod api;
mod embedder;
#[cfg(feature = "onnx")]
mod onnx;

pub use api::ApiEmbedding;
pub use embedder::Embedder;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedding;
