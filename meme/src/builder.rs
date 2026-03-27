//! [`MemeBuilder`] — fluent builder for constructing a [`Meme`](crate::Meme) instance.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::config::{self, Config};
use crate::embedding::{self, Embedder};
use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::meme::Meme;
use crate::pipeline::{Extractor, HybridRetriever};
use crate::store::{HistoryStore, VectorStore};

/// Fluent builder for constructing a [`Meme`] instance.
///
/// All settings have sensible defaults. Only [`api_key`](Self::api_key) is
/// required for the default API-based embedding provider.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example() -> meme::error::Result<()> {
/// let meme = meme::Meme::builder()
///     .api_key("sk-...")
///     .model("gpt-4.1-mini")
///     .namespace("user-42")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// Load from a TOML file and override specific fields:
///
/// ```rust,no_run
/// # async fn example(config: meme::config::Config) -> meme::error::Result<()> {
/// let meme = meme::Meme::builder()
///     .config(config)
///     .api_key("override-key")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct MemeBuilder {
    config: Config,
    http_client: Option<reqwest::Client>,
    clear_db: bool,
    namespace: Option<String>,
}

impl MemeBuilder {
    /// Create a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the entire configuration (e.g. loaded from a TOML file).
    ///
    /// Subsequent setter calls override individual fields on top of this.
    #[must_use]
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Set the LLM API key.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.llm.api_key = Some(key.into());
        self
    }

    /// Set the LLM model name (e.g. `"gpt-4.1-mini"`).
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.llm.model = model.into();
        self
    }

    /// Set the LLM base URL (OpenAI-compatible endpoint).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.config.llm.base_url = url.into();
        self
    }

    /// Set the embedding model name.
    #[must_use]
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.config.embedding.model = model.into();
        self
    }

    /// Set the embedding vector dimension.
    #[must_use]
    pub const fn embedding_dimension(mut self, dim: usize) -> Self {
        self.config.embedding.dimension = dim;
        self
    }

    /// Set the `LanceDB` storage directory path.
    #[must_use]
    pub fn lancedb_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.store.lancedb_path = path.into();
        self
    }

    /// Set the `SQLite` history database file path.
    #[must_use]
    pub fn history_db_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.store.history_db_path = path.into();
        self
    }

    /// Enable or disable LLM-driven query planning.
    #[must_use]
    pub const fn enable_planning(mut self, enable: bool) -> Self {
        self.config.pipeline.enable_planning = enable;
        self
    }

    /// Enable or disable reflection-based retrieval refinement.
    #[must_use]
    pub const fn enable_reflection(mut self, enable: bool) -> Self {
        self.config.pipeline.enable_reflection = enable;
        self
    }

    /// Set the number of results for semantic (ANN) search.
    #[must_use]
    pub const fn semantic_top_k(mut self, k: usize) -> Self {
        self.config.pipeline.semantic_top_k = k;
        self
    }

    /// Enable local ONNX reranker with the given model name.
    ///
    /// Requires the `onnx` feature. Example model: `"BAAI/bge-reranker-v2-m3"`.
    #[must_use]
    pub fn reranker(mut self, model: impl Into<String>) -> Self {
        self.config.pipeline.reranker_model = Some(model.into());
        self
    }

    /// Set the number of top results to keep after reranking.
    #[must_use]
    pub const fn rerank_top_n(mut self, n: usize) -> Self {
        self.config.pipeline.rerank_top_n = n;
        self
    }

    /// Provide a pre-configured [`reqwest::Client`].
    ///
    /// Use this to customize timeouts, proxies, TLS certificates, or
    /// connection pooling. When omitted, a default client is created.
    #[must_use]
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Clear the database on initialization.
    #[must_use]
    pub const fn clear_db(mut self, clear: bool) -> Self {
        self.clear_db = clear;
        self
    }

    /// Set the namespace for memory isolation.
    ///
    /// The library treats this as an opaque string — callers decide the
    /// semantics (user ID, session ID, composite key, etc.).
    #[must_use]
    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    /// Build the [`Meme`] instance.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or storage cannot be initialized.
    pub async fn build(self) -> Result<Meme> {
        let config = self.config;
        config.validate()?;

        let http = self.http_client.map_or_else(build_http_client, Ok)?;
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
                return Err(Error::Config(
                    "ONNX provider requires the 'onnx' feature flag".into(),
                ));
            }
        });

        let store = Arc::new(
            VectorStore::open(
                &config.store.lancedb_path.to_string_lossy(),
                &config.store.table_name,
                embedder.dimension(),
            )
            .await?,
        );

        let history = Arc::new(HistoryStore::open(&config.store.history_db_path)?);

        if self.clear_db {
            store.clear_all().await?;
        }

        let extractor = Extractor::new(
            Arc::clone(&llm),
            &config.pipeline,
            config.pipeline.max_build_workers,
        );

        #[cfg(feature = "onnx")]
        let reranker = config
            .pipeline
            .reranker_model
            .as_deref()
            .map(crate::reranking::OnnxReranker::new)
            .transpose()?
            .map(Arc::new);

        #[cfg(not(feature = "onnx"))]
        if config.pipeline.reranker_model.is_some() {
            return Err(Error::Config(
                "reranker requires the 'onnx' feature flag".into(),
            ));
        }

        let retriever = HybridRetriever::new(
            Arc::clone(&llm),
            Arc::clone(&store),
            Arc::clone(&embedder),
            config.pipeline.clone(),
            self.namespace.clone(),
            #[cfg(feature = "onnx")]
            reranker,
        );

        tracing::info!("meme system initialized");

        Ok(Meme {
            llm,
            store,
            embedder,
            history,
            extractor: Mutex::new(extractor),
            retriever,
            config,
            namespace: self.namespace,
        })
    }
}

/// Default request timeout.
const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(1);

/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Default max idle connections per host.
const DEFAULT_POOL_IDLE_PER_HOST: usize = 10;

/// Build a shared [`reqwest::Client`] with production-ready defaults.
fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(DEFAULT_TIMEOUT)
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .pool_max_idle_per_host(DEFAULT_POOL_IDLE_PER_HOST)
        .user_agent(concat!("meme/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| Error::Internal(format!("failed to build HTTP client: {e}")))
}
