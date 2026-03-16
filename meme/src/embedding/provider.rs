//! Trait definition for embedding providers.

use crate::error::Result;

/// Unified interface for embedding vector computation.
///
/// Implementations may use a remote API or a local ONNX model.
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Returns the dimensionality of the embedding vectors.
    fn dimension(&self) -> usize;

    /// Encode a batch of document texts into embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Encode a single query text into an embedding vector.
    ///
    /// Query encoding may use a different prompt prefix than document encoding
    /// (e.g., Qwen3's `query:` prefix for asymmetric retrieval).
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    async fn encode_query(&self, text: &str) -> Result<Vec<f32>>;
}
