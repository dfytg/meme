//! API-based embedding provider — calls an OpenAI-compatible embeddings endpoint.

use std::time::Duration;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Maximum texts per single API call to avoid timeouts on large batches.
const EMBED_BATCH_SIZE: usize = 128;

/// Embedding provider that calls a remote OpenAI-compatible API.
#[derive(Debug, Clone)]
pub struct ApiEmbedding {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    dimension: usize,
    max_retries: u32,
}

impl ApiEmbedding {
    /// Create a new API embedding provider using a shared HTTP client.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is missing.
    pub fn new(
        http: reqwest::Client,
        embedding_cfg: &crate::config::EmbeddingConfig,
        llm_cfg: &crate::config::LlmConfig,
    ) -> Result<Self> {
        let api_key = llm_cfg
            .api_key
            .clone()
            .ok_or_else(|| Error::Config("API key is required for API embedding".to_owned()))?;
        Ok(Self {
            http,
            base_url: llm_cfg.base_url.trim_end_matches('/').to_owned(),
            api_key,
            model: embedding_cfg.model.clone(),
            dimension: embedding_cfg.dimension,
            max_retries: llm_cfg.max_retries,
        })
    }

    /// Returns the dimensionality of the embedding vectors.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Encode a batch of document texts into embedding vectors.
    ///
    /// Large batches are automatically chunked to avoid API timeouts.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    #[tracing::instrument(skip(self, texts), fields(count = texts.len()))]
    pub async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() <= EMBED_BATCH_SIZE {
            let input: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
            return self.embed_with_retry(input).await;
        }

        // Parallel chunk embedding with concurrency limit.
        let chunks: Vec<Vec<String>> = texts
            .chunks(EMBED_BATCH_SIZE)
            .map(|chunk| chunk.iter().map(|s| (*s).to_owned()).collect())
            .collect();

        let max_concurrent = 4;
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let this = self.clone();
            let sem = std::sync::Arc::clone(&semaphore);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await;
                this.embed_with_retry(chunk).await
            }));
        }

        let mut all_vectors = Vec::with_capacity(texts.len());
        for handle in handles {
            let vectors = handle
                .await
                .map_err(|e| Error::Embedding(format!("embed task panicked: {e}")))??;
            all_vectors.extend(vectors);
        }
        Ok(all_vectors)
    }

    /// Encode a single query text into an embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub async fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_with_retry(vec![text.to_owned()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty embedding response for query".to_owned()))
    }

    async fn embed_with_retry(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut last_err = None;
        for attempt in 0..self.max_retries {
            match self.call_api(&input).await {
                Ok(vectors) => return Ok(vectors),
                Err(e) => {
                    tracing::warn!(attempt = attempt + 1, error = %e, "embedding API call failed");
                    last_err = Some(e);
                    if attempt + 1 < self.max_retries {
                        let wait = 1u64 << attempt;
                        tokio::time::sleep(Duration::from_secs(wait)).await;
                    }
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| Error::Embedding("all embedding retries exhausted".to_owned())))
    }

    async fn call_api(&self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "input": input,
            "dimensions": self.dimension,
        });

        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Embedding(format!(
                "embedding API returned {status}: {text}"
            )));
        }

        let data: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| Error::Embedding(format!("failed to parse embedding response: {e}")))?;

        let mut vectors: Vec<(usize, Vec<f32>)> = data
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        vectors.sort_by_key(|(idx, _)| *idx);

        Ok(vectors.into_iter().map(|(_, v)| v).collect())
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}
