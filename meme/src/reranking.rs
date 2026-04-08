//! Local ONNX reranker via [`fastembed`].
//!
//! Requires the `onnx` feature flag. Models are downloaded automatically
//! from Hugging Face Hub on first use.
//!
//! The reranker re-scores retrieval results using a cross-encoder model,
//! which jointly attends to the query and each document for fine-grained
//! relevance estimation — significantly more accurate than bi-encoder
//! cosine similarity alone.

use std::sync::{Arc, Mutex};

use crate::error::{MemeError, Result};

/// Local reranker powered by [`fastembed::TextRerank`].
///
/// Wraps an ONNX cross-encoder model that scores `(query, document)` pairs.
/// Thread-safe via `Arc<Mutex<_>>` and async-safe via `spawn_blocking`.
pub(crate) struct OnnxReranker {
    /// Thread-safe handle to the ONNX reranker model.
    model: Arc<Mutex<fastembed::TextRerank>>,
}

impl std::fmt::Debug for OnnxReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxReranker").finish_non_exhaustive()
    }
}

impl OnnxReranker {
    /// Create a new reranker.
    ///
    /// `model_name` must match a fastembed reranker model code
    /// (e.g. `"BAAI/bge-reranker-v2-m3"`).
    /// The model is downloaded automatically on first use.
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is unknown or initialization fails.
    pub(crate) fn new(model_name: &str) -> Result<Self> {
        let reranker_model = resolve_model(model_name)?;
        let model = fastembed::TextRerank::try_new(
            fastembed::RerankInitOptions::new(reranker_model).with_show_download_progress(true),
        )
        .map_err(|e| MemeError::Internal(format!("reranker init failed: {e}")))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }

    /// Re-rank documents by relevance to `query` and return the top-N indices.
    ///
    /// Returns indices into the original `documents` slice, ordered by
    /// descending relevance score. At most `top_n` indices are returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model inference fails.
    pub(crate) async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> Result<Vec<usize>> {
        if documents.is_empty() || top_n == 0 {
            return Ok(Vec::new());
        }

        let model = Arc::clone(&self.model);
        let query = query.to_owned();
        let docs: Vec<String> = documents.iter().map(|s| (*s).to_owned()).collect();
        let n = top_n.min(docs.len());

        tokio::task::spawn_blocking(move || {
            let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            let results = {
                let mut guard = model
                    .lock()
                    .map_err(|e| MemeError::Internal(format!("reranker lock poisoned: {e}")))?;
                guard
                    .rerank(query.as_str(), doc_refs.as_slice(), false, None)
                    .map_err(|e| MemeError::Internal(format!("reranker inference failed: {e}")))?
            };

            Ok(results.into_iter().take(n).map(|r| r.index).collect())
        })
        .await
        .map_err(|e| MemeError::Internal(format!("reranker spawn_blocking failed: {e}")))?
    }
}

/// Resolve a model code string to a [`fastembed::RerankerModel`].
fn resolve_model(name: &str) -> Result<fastembed::RerankerModel> {
    for info in fastembed::TextRerank::list_supported_models() {
        if info.model_code == name {
            return Ok(info.model);
        }
    }
    Err(MemeError::Config(format!("unknown reranker model: {name}")))
}
