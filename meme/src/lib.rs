//! # meme
//!
//! Long-term memory for AI agents.
//!
//! A Rust implementation of the `SimpleMem` three-stage pipeline:
//! 1. **Semantic Structured Compression** — dialogues → compact memory entries
//! 2. **Online Semantic Synthesis** — deduplication during write
//! 3. **Intent-Aware Retrieval Planning** — multi-view hybrid retrieval
//!
//! Memory is persistent across sessions — the vector store is stored on disk.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use meme::{Meme, MemeBuilder};
//!
//! # async fn example() -> meme::error::Result<()> {
//! let meme = MemeBuilder::new()
//!     .api_key("sk-...")
//!     .model("gpt-4.1-mini")
//!     .build()
//!     .await?;
//!
//! meme.add_dialogue("Alice", "Let's meet at 2pm tomorrow", None).await?;
//! meme.finalize().await?;
//!
//! let answer = meme.ask("When will Alice meet?").await?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]

pub mod config;
pub mod embedding;
pub mod error;
pub mod llm;
pub mod model;
pub mod pipeline;
pub mod store;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use config::Config;
use embedding::{ApiEmbedding, Embedder};
use error::Result;
use llm::LlmClient;
use model::{Dialogue, MemoryEntry};
use pipeline::{HybridRetriever, MemoryBuilder};
use store::VectorStore;
use tokio::sync::Mutex;

/// The main entry point for the meme memory system.
///
/// Wraps the three-stage pipeline (compression, synthesis, retrieval)
/// behind a simple async API.
pub struct Meme {
    llm: Arc<LlmClient>,
    store: Arc<VectorStore>,
    embedder: Arc<Embedder>,
    builder: Mutex<MemoryBuilder>,
    retriever: HybridRetriever,
    config: Config,
}

impl std::fmt::Debug for Meme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Meme")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[allow(clippy::future_not_send)]
impl Meme {
    /// Create a builder for configuring a new `Meme` instance.
    #[must_use]
    pub fn builder() -> MemeBuilder {
        MemeBuilder::new()
    }

    /// Add a single dialogue turn.
    ///
    /// When the internal buffer reaches `window_size`, entries are automatically
    /// extracted and stored.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction or storage fails.
    pub async fn add_dialogue(
        &self,
        speaker: &str,
        content: &str,
        timestamp: Option<DateTime<Utc>>,
    ) -> Result<()> {
        let mut dialogue = Dialogue::new(speaker, content);
        if let Some(ts) = timestamp {
            dialogue = dialogue.with_timestamp(ts);
        }

        let mut builder = self.builder.lock().await;
        let entries = builder.add_dialogue(dialogue).await?;
        if !entries.is_empty() {
            self.store_entries(&entries).await?;
        }
        Ok(())
    }

    /// Batch add dialogues.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction or storage fails.
    pub async fn add_dialogues(&self, dialogues: Vec<Dialogue>) -> Result<()> {
        let mut builder = self.builder.lock().await;
        let entries = builder.add_dialogues(dialogues).await?;
        if !entries.is_empty() {
            self.store_entries(&entries).await?;
        }
        Ok(())
    }

    /// Process any remaining dialogues in the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction or storage fails.
    pub async fn finalize(&self) -> Result<()> {
        let mut builder = self.builder.lock().await;
        let entries = builder.finalize().await?;
        if !entries.is_empty() {
            self.store_entries(&entries).await?;
        }
        Ok(())
    }

    /// Ask a question — the core Q&A interface.
    ///
    /// Executes intent-aware retrieval planning, multi-view hybrid search,
    /// and generates a concise answer.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval or answer generation fails.
    pub async fn ask(&self, question: &str) -> Result<String> {
        tracing::info!(question, "processing question");
        let contexts = self.retriever.retrieve(question).await?;
        let answer = pipeline::generator::generate(&self.llm, question, &contexts).await?;
        tracing::info!(question, answer = answer.as_str(), "answer generated");
        Ok(answer)
    }

    /// Get all stored memory entries (for debugging).
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn get_all_memories(&self) -> Result<Vec<MemoryEntry>> {
        self.store.get_all().await
    }

    /// Count stored memory entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn memory_count(&self) -> Result<usize> {
        self.store.count().await
    }

    /// Clear all stored memories.
    ///
    /// # Errors
    ///
    /// Returns an error if the clear operation fails.
    pub async fn clear(&self) -> Result<()> {
        self.store.clear().await
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    async fn store_entries(&self, entries: &[MemoryEntry]) -> Result<()> {
        let texts: Vec<&str> = entries.iter().map(|e| e.restatement.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;
        self.store.add_entries(entries, &vectors).await
    }
}

/// Builder for constructing a [`Meme`] instance.
#[derive(Debug, Default)]
pub struct MemeBuilder {
    config: Option<Config>,
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    clear_db: bool,
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

    /// Build the `Meme` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or storage cannot be initialized.
    pub async fn build(self) -> Result<Meme> {
        let mut config = self
            .config
            .unwrap_or_else(|| Config::load_default().unwrap_or_default());

        // Apply overrides.
        if let Some(key) = self.api_key {
            config.llm.api_key = Some(key);
        }
        if let Some(model) = self.model {
            config.llm.model = model;
        }
        if let Some(url) = self.base_url {
            config.llm.base_url = url;
        }

        // Build components.
        let llm = Arc::new(LlmClient::from_config(&config.llm)?);

        let embedder = Arc::new(match config.embedding.provider {
            config::EmbeddingProviderKind::Api => {
                Embedder::Api(ApiEmbedding::from_config(&config.embedding, &config.llm)?)
            }
            #[cfg(feature = "onnx")]
            config::EmbeddingProviderKind::Onnx => {
                let model_path = config.embedding.onnx_model_path.as_deref().ok_or_else(|| {
                    error::Error::Config("onnx_model_path is required for ONNX provider".into())
                })?;
                let tokenizer_path =
                    config
                        .embedding
                        .onnx_tokenizer_path
                        .as_deref()
                        .ok_or_else(|| {
                            error::Error::Config(
                                "onnx_tokenizer_path is required for ONNX provider".into(),
                            )
                        })?;
                Embedder::Onnx(embedding::OnnxEmbedding::from_paths(
                    model_path,
                    tokenizer_path,
                    config.embedding.dimension,
                )?)
            }
            #[cfg(not(feature = "onnx"))]
            config::EmbeddingProviderKind::Onnx => {
                return Err(error::Error::Config(
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

        if self.clear_db {
            store.clear().await?;
        }

        let mem_builder = MemoryBuilder::new(
            Arc::clone(&llm),
            &config.pipeline,
            config.pipeline.max_build_workers,
        );

        let retriever = HybridRetriever::new(
            Arc::clone(&llm),
            Arc::clone(&store),
            Arc::clone(&embedder),
            &config.pipeline,
            config.pipeline.max_retrieval_workers,
        );

        tracing::info!("meme system initialized");

        Ok(Meme {
            llm,
            store,
            embedder,
            builder: Mutex::new(mem_builder),
            retriever,
            config,
        })
    }
}
