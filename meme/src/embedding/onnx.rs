//! Local ONNX embedding via [`fastembed`].
//!
//! Requires the `onnx` feature flag. Models are downloaded automatically
//! from Hugging Face Hub on first use.

use std::sync::{Arc, Mutex};

use crate::error::{MemeError, Result};

/// Local embedding provider powered by [`fastembed`].
///
/// Handles model download, tokenization, ONNX inference, pooling, and
/// L2 normalization automatically.
pub(crate) struct OnnxEmbedding {
    model: Arc<Mutex<fastembed::TextEmbedding>>,
    dimension: usize,
}

impl std::fmt::Debug for OnnxEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbedding")
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl OnnxEmbedding {
    /// Create a new local embedding provider.
    ///
    /// `model_name` must match a fastembed model code
    /// (e.g. `"BAAI/bge-small-en-v1.5"`).
    /// The model is downloaded automatically on first use.
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is unknown or initialization fails.
    pub(crate) fn new(model_name: &str) -> Result<Self> {
        let (embedding_model, dimension) = resolve_model(model_name)?;
        let model = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(embedding_model).with_show_download_progress(true),
        )
        .map_err(|e| MemeError::Embedding(format!("fastembed init failed: {e}")))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            dimension,
        })
    }

    /// Returns the embedding dimension.
    #[must_use]
    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Encode a batch of document texts into embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub(crate) async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
        let model = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || {
            let mut guard = model
                .lock()
                .map_err(|e| MemeError::Embedding(format!("fastembed lock poisoned: {e}")))?;
            guard
                .embed(owned, None)
                .map_err(|e| MemeError::Embedding(format!("fastembed encode failed: {e}")))
        })
        .await
        .map_err(|e| MemeError::Embedding(format!("spawn_blocking failed: {e}")))?
    }

    /// Encode a single query text into an embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub(crate) async fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.encode_documents(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| MemeError::Embedding("empty fastembed result".to_owned()))
    }
}

fn resolve_model(name: &str) -> Result<(fastembed::EmbeddingModel, usize)> {
    for info in fastembed::TextEmbedding::list_supported_models() {
        if info.model_code == name {
            return Ok((info.model, info.dim));
        }
    }
    Err(MemeError::Config(format!("unknown fastembed model: {name}")))
}
