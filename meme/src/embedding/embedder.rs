//! Embedding provider — unified enum dispatch for API and ONNX backends.

use super::api::ApiEmbedding;
use crate::error::Result;

/// Unified embedding provider using enum dispatch (zero-cost, no boxing).
#[derive(Debug)]
pub(crate) enum Embedder {
    /// Remote API-based embedding.
    Api(ApiEmbedding),
    /// Local ONNX Runtime inference.
    #[cfg(feature = "onnx")]
    Onnx(super::onnx::OnnxEmbedding),
}

impl Embedder {
    /// Returns the dimensionality of the embedding vectors.
    #[must_use]
    pub(crate) const fn dimension(&self) -> usize {
        match self {
            Self::Api(e) => e.dimension(),
            #[cfg(feature = "onnx")]
            Self::Onnx(e) => e.dimension(),
        }
    }

    /// Encode a batch of document texts into embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub(crate) async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Api(e) => e.encode_documents(texts).await,
            #[cfg(feature = "onnx")]
            Self::Onnx(e) => e.encode_documents(texts).await,
        }
    }

    /// Encode a single query text into an embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub(crate) async fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Api(e) => e.encode_query(text).await,
            #[cfg(feature = "onnx")]
            Self::Onnx(e) => e.encode_query(text).await,
        }
    }
}
