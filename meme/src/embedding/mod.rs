//! Embedding model abstraction — unified interface for API and local ONNX providers.

mod api;
#[cfg(feature = "onnx")]
mod onnx;
mod provider;

pub use api::ApiEmbedding;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedding;
pub use provider::EmbeddingProvider;
