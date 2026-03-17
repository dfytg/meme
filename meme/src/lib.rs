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
pub mod http;
pub mod llm;
pub mod model;
pub mod pipeline;
pub mod store;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use config::Config;
use embedding::Embedder;
use error::Result;
use llm::LlmClient;
use model::{Dialogue, MemoryEntry};
use pipeline::{HybridRetriever, MemoryBuilder};
use store::{Scope, VectorStore};
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
    scope: Scope,
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
    #[tracing::instrument(skip(self, content, timestamp), fields(speaker))]
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

    /// Import pre-existing memory entries by recomputing embeddings and storing them.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding computation or storage fails.
    #[tracing::instrument(skip(self, entries), fields(count = entries.len()))]
    pub async fn import_entries(&self, entries: &mut [MemoryEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        for entry in entries.iter_mut() {
            if entry.user_id.is_none() {
                entry.user_id.clone_from(&self.scope.user_id);
            }
            if entry.session_id.is_none() {
                entry.session_id.clone_from(&self.scope.session_id);
            }
        }
        self.store_entries(entries).await
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
    #[tracing::instrument(skip(self))]
    pub async fn ask(&self, question: &str) -> Result<String> {
        let contexts = self.retriever.retrieve(question).await?;
        let answer = pipeline::generator::generate(&self.llm, question, &contexts).await?;
        tracing::info!(contexts = contexts.len(), "answer generated");
        Ok(answer)
    }

    /// Get all stored memory entries (for debugging).
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn get_all_memories(&self) -> Result<Vec<MemoryEntry>> {
        self.store.get_all(&self.scope).await
    }

    /// Count stored memory entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn memory_count(&self) -> Result<usize> {
        self.store.count(&self.scope).await
    }

    /// Clear stored memories for the current scope.
    ///
    /// If a user/session scope is set, only that scope's entries are removed.
    /// If no scope is set, **all** entries are removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the clear operation fails.
    pub async fn clear(&self) -> Result<()> {
        self.store.clear(&self.scope).await
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    async fn store_entries(&self, entries: &[MemoryEntry]) -> Result<()> {
        let mut scoped: Vec<MemoryEntry> = entries.to_vec();
        for entry in &mut scoped {
            if entry.user_id.is_none() {
                entry.user_id.clone_from(&self.scope.user_id);
            }
            if entry.session_id.is_none() {
                entry.session_id.clone_from(&self.scope.session_id);
            }
        }

        let texts: Vec<&str> = scoped.iter().map(|e| e.restatement.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;

        let (accepted, accepted_vecs, ids_to_delete) =
            self.resolve_conflicts(&scoped, &vectors).await?;

        if !ids_to_delete.is_empty() {
            self.store.delete_entries(&ids_to_delete).await?;
            tracing::info!(count = ids_to_delete.len(), "deleted superseded memories");
        }

        if !accepted.is_empty() {
            self.store.add_entries(&accepted, &accepted_vecs).await?;
        }
        Ok(())
    }

    /// Check each new entry against existing similar memories and resolve conflicts.
    ///
    /// Returns (entries to insert, their vectors, IDs to delete from store).
    async fn resolve_conflicts(
        &self,
        entries: &[MemoryEntry],
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<MemoryEntry>, Vec<Vec<f32>>, Vec<String>)> {
        let mut accepted = Vec::new();
        let mut accepted_vecs = Vec::new();
        let mut ids_to_delete = Vec::new();

        let existing_count = self.store.count(&self.scope).await?;
        if existing_count == 0 {
            return Ok((entries.to_vec(), vectors.to_vec(), Vec::new()));
        }

        let similarity_top_k = 3;
        for (i, entry) in entries.iter().enumerate() {
            let similar = self
                .store
                .semantic_search(&vectors[i], similarity_top_k, &self.scope)
                .await?;

            if similar.is_empty() {
                accepted.push(entry.clone());
                accepted_vecs.push(vectors[i].clone());
                continue;
            }

            let existing_pairs: Vec<(usize, &str)> = similar
                .iter()
                .enumerate()
                .map(|(j, e)| (j, e.restatement.as_str()))
                .collect();

            match self
                .resolve_single_conflict(&entry.restatement, &existing_pairs)
                .await
            {
                Ok(actions) => {
                    let mut is_duplicate = false;
                    for action in &actions {
                        let act = action["action"].as_str().unwrap_or("keep_both");
                        let idx = action["existing_index"].as_u64().unwrap_or(u64::MAX) as usize;
                        match act {
                            "duplicate" => {
                                is_duplicate = true;
                                tracing::info!(
                                    new = entry.restatement.as_str(),
                                    "skipping duplicate memory"
                                );
                            }
                            "update" if idx < similar.len() => {
                                ids_to_delete.push(similar[idx].id.to_string());
                            }
                            _ => {}
                        }
                    }
                    if !is_duplicate {
                        accepted.push(entry.clone());
                        accepted_vecs.push(vectors[i].clone());
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "conflict resolution failed, keeping entry");
                    accepted.push(entry.clone());
                    accepted_vecs.push(vectors[i].clone());
                }
            }
        }

        Ok((accepted, accepted_vecs, ids_to_delete))
    }

    async fn resolve_single_conflict(
        &self,
        new_text: &str,
        existing: &[(usize, &str)],
    ) -> Result<Vec<serde_json::Value>> {
        let prompt_text = llm::prompt::conflict_resolution(new_text, existing);
        let messages = vec![
            llm::Message::system(
                "You are a memory conflict resolution assistant. You must output valid JSON format.",
            ),
            llm::Message::user(prompt_text),
        ];
        let opts = llm::ChatOptions {
            temperature: 0.1,
            json_mode: true,
        };
        let response = self.llm.chat(&messages, &opts).await?;
        let data = llm::extract_json_from_text(&response)?;
        Ok(data.as_array().cloned().unwrap_or_default())
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
            builder: Mutex::new(mem_builder),
            retriever,
            config,
            scope,
        })
    }
}
