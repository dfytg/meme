//! Local ONNX Runtime embedding provider.
//!
//! Requires the `onnx` feature flag to be enabled.

use std::path::Path;

use super::provider::EmbeddingProvider;
use crate::error::{Error, Result};

/// Embedding provider that runs a local ONNX model via `ort`.
pub struct OnnxEmbedding {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
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
    /// Load an ONNX model and tokenizer from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer files cannot be loaded.
    pub fn from_paths(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        dimension: usize,
    ) -> Result<Self> {
        let session = ort::session::Session::builder()
            .and_then(|b| b.with_model_from_file(model_path.as_ref()))
            .map_err(|e| Error::Embedding(format!("failed to load ONNX model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Embedding(format!("failed to load tokenizer: {e}")))?;

        Ok(Self {
            session,
            tokenizer,
            dimension,
        })
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Embedding(format!("tokenization failed: {e}")))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let batch_size = texts.len();
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[i * max_len + j] = i64::from(id);
                attention_mask[i * max_len + j] = i64::from(m);
            }
        }

        let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, max_len), input_ids)
            .map_err(|e| Error::Embedding(format!("shape error: {e}")))?;
        let attention_mask_array =
            ndarray::Array2::from_shape_vec((batch_size, max_len), attention_mask)
                .map_err(|e| Error::Embedding(format!("shape error: {e}")))?;

        let outputs = self
            .session
            .run(
                ort::inputs![input_ids_array, attention_mask_array]
                    .map_err(|e| Error::Embedding(format!("input creation failed: {e}")))?,
            )
            .map_err(|e| Error::Embedding(format!("ONNX inference failed: {e}")))?;

        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(format!("output extraction failed: {e}")))?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let embedding: Vec<f32> = output_tensor
                .slice(ndarray::s![i, ..self.dimension])
                .to_vec();
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized = if norm > 0.0 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding
            };
            results.push(normalized);
        }

        Ok(results)
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OnnxEmbedding {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.encode_batch(texts)
    }

    async fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.encode_batch(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty result from ONNX query encoding".to_owned()))
    }
}
