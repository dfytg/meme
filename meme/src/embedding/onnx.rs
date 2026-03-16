//! Local ONNX Runtime embedding provider.
//!
//! Requires the `onnx` feature flag to be enabled.
//! Built against `ort 2.0.0-rc.12` API.

use std::path::Path;
use std::sync::{Arc, Mutex};

use super::provider::EmbeddingProvider;
use crate::error::{Error, Result};

/// Embedding provider that runs a local ONNX model via `ort`.
///
/// Session is wrapped in `Arc<Mutex<_>>` because `Session::run` requires
/// `&mut self`. Tokenizer is in `Arc` for cheap cloning into blocking tasks.
pub struct OnnxEmbedding {
    session: Arc<Mutex<ort::session::Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
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
            .and_then(|mut b| b.commit_from_file(model_path.as_ref()))
            .map_err(|e| Error::Embedding(format!("failed to load ONNX model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Embedding(format!("failed to load tokenizer: {e}")))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            dimension,
        })
    }
}

fn encode_batch_sync(
    session: &Mutex<ort::session::Session>,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[&str],
    dimension: usize,
) -> Result<Vec<Vec<f32>>> {
    let encodings = tokenizer
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

    let ids_tensor = ort::value::TensorRef::from_array_view(&input_ids_array)
        .map_err(|e| Error::Embedding(format!("tensor creation failed: {e}")))?;
    let mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask_array)
        .map_err(|e| Error::Embedding(format!("tensor creation failed: {e}")))?;

    let mut sess = session
        .lock()
        .map_err(|e| Error::Embedding(format!("session lock poisoned: {e}")))?;

    let outputs = sess
        .run(ort::inputs![ids_tensor, mask_tensor])
        .map_err(|e| Error::Embedding(format!("ONNX inference failed: {e}")))?;

    let (_shape, flat_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Embedding(format!("output extraction failed: {e}")))?;

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let start = i * dimension;
        let end = start + dimension;
        let embedding = if end <= flat_data.len() {
            flat_data[start..end].to_vec()
        } else {
            vec![0.0f32; dimension]
        };
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

#[async_trait::async_trait]
impl EmbeddingProvider for OnnxEmbedding {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let dimension = self.dimension;
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            encode_batch_sync(&session, &tokenizer, &refs, dimension)
        })
        .await
        .map_err(|e| Error::Embedding(format!("spawn_blocking failed: {e}")))?
    }

    async fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.encode_documents(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty result from ONNX query encoding".to_owned()))
    }
}
