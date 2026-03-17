//! API-based embedding provider — calls an OpenAI-compatible embeddings endpoint.

use serde::Deserialize;

use crate::error::{Error, Result};

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
    /// Create a new API embedding provider.
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            dimension,
            max_retries: 3,
        }
    }

    /// Create from the embedding config section.
    ///
    /// # Errors
    ///
    /// Returns an error if the LLM API key is missing.
    pub fn from_config(
        embedding_cfg: &crate::config::EmbeddingConfig,
        llm_cfg: &crate::config::LlmConfig,
    ) -> Result<Self> {
        let api_key = llm_cfg
            .api_key
            .clone()
            .ok_or_else(|| Error::Config("API key is required for API embedding".to_owned()))?;
        Ok(Self::new(
            api_key,
            &embedding_cfg.model,
            &llm_cfg.base_url,
            embedding_cfg.dimension,
        ))
    }

    /// Returns the dimensionality of the embedding vectors.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Encode a batch of document texts into embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub async fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let input: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
        self.embed(input).await
    }

    /// Encode a single query text into an embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub async fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed(vec![text.to_owned()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty embedding response for query".to_owned()))
    }

    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut last_err = None;
        for attempt in 0..self.max_retries {
            match self.call_embed_api(&input).await {
                Ok(vectors) => return Ok(vectors),
                Err(e) => {
                    tracing::warn!(attempt = attempt + 1, error = %e, "embedding API call failed");
                    last_err = Some(e);
                    if attempt + 1 < self.max_retries {
                        let wait = 1u64 << attempt;
                        tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    }
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| Error::Embedding("all embedding retries exhausted".to_owned())))
    }

    async fn call_embed_api(&self, input: &[String]) -> Result<Vec<Vec<f32>>> {
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
            .header("Content-Type", "application/json")
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
