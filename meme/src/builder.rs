//! [`MemeBuilder`] — fluent builder for constructing a [`Meme`](crate::Meme) instance.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::config::{self, Config};
use crate::embedding::{self, Embedder};
use crate::error::Result;
use crate::facade::Meme;
use crate::http;
use crate::llm::LlmClient;
use crate::pipeline::{HybridRetriever, MemoryBuilder};
use crate::store::{HistoryStore, Scope, VectorStore};

/// Builder for constructing a [`Meme`] instance.
#[derive(Debug, Default)]
pub struct MemeBuilder {
    config: Option<Config>,
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    clear_db: bool,
    user_id: Option<String>,
    session_id: Option<String>,
}

impl MemeBuilder {
    /// Create a new builder with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a full configuration.
    #[must_use]
    pub fn config(mut self, config: Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the LLM API key (overrides config).
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the LLM model name (overrides config).
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the LLM base URL (overrides config).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Clear the database on initialization.
    #[must_use]
    pub const fn clear_db(mut self, clear: bool) -> Self {
        self.clear_db = clear;
        self
    }

    /// Set the user identifier for multi-tenant isolation.
    #[must_use]
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Set the session identifier for multi-session isolation.
    #[must_use]
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Build the `Meme` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or storage cannot be initialized.
    pub async fn build(self) -> Result<Meme> {
        let mut config = self
            .config
            .unwrap_or_else(|| Config::load_default().unwrap_or_default());

        if let Some(key) = self.api_key {
            config.llm.api_key = Some(key);
        }
        if let Some(model) = self.model {
            config.llm.model = model;
        }
        if let Some(url) = self.base_url {
            config.llm.base_url = url;
        }

        config.validate()?;

        let http = http::build_http_client()?;
        let llm = Arc::new(LlmClient::new(http.clone(), &config.llm)?);

        let embedder = Arc::new(match config.embedding.provider {
            config::EmbeddingProviderKind::Api => Embedder::Api(embedding::ApiEmbedding::new(
                http,
                &config.embedding,
                &config.llm,
            )?),
            #[cfg(feature = "onnx")]
            config::EmbeddingProviderKind::Onnx => {
                Embedder::Onnx(embedding::OnnxEmbedding::new(&config.embedding.model)?)
            }
            #[cfg(not(feature = "onnx"))]
            config::EmbeddingProviderKind::Onnx => {
                return Err(crate::error::Error::Config(
                    "ONNX provider requires the 'onnx' feature flag".into(),
                ));
            }
        });

        let store = Arc::new(
            VectorStore::open(
                &config.store.lancedb_path,
                &config.store.table_name,
                embedder.dimension(),
            )
            .await?,
        );

        let history_path = std::path::Path::new(&config.store.history_db_path);
        let history = Arc::new(HistoryStore::open(history_path)?);

        if self.clear_db {
            store.clear_all().await?;
        }

        let mem_builder = MemoryBuilder::new(
            Arc::clone(&llm),
            &config.pipeline,
            config.pipeline.max_build_workers,
        );

        let scope = Scope {
            user_id: self.user_id,
            session_id: self.session_id,
        };

        let retriever = HybridRetriever::new(
            Arc::clone(&llm),
            Arc::clone(&store),
            Arc::clone(&embedder),
            &config.pipeline,
            config.pipeline.max_retrieval_workers,
            scope.clone(),
        );

        tracing::info!("meme system initialized");

        Ok(Meme {
            llm,
            store,
            embedder,
            history,
            builder: Mutex::new(mem_builder),
            retriever,
            config,
            scope,
        })
    }
}
