//! [`Meme`] — the public-facing API for the memory system.
//!
//! Provides CRUD operations, dialogue ingestion, Q&A, and lifecycle
//! reconciliation behind a single ergonomic struct.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::Config;
use crate::embedding::Embedder;
use crate::error::{Error, Result};
use crate::llm::{self, LlmClient, ReExtractResponse};
use crate::model::{Dialogue, Event, EventType, Memory, Scope};
use crate::pipeline::{self, HybridRetriever, MemoryBuilder};
use crate::store::{HistoryStore, VectorStore};

/// The main entry point for the meme memory system.
///
/// Wraps the three-stage pipeline (compression, reconciliation, retrieval)
/// behind a simple async API with full CRUD and history tracking.
pub struct Meme {
    pub(crate) llm: Arc<LlmClient>,
    pub(crate) store: Arc<VectorStore>,
    pub(crate) embedder: Arc<Embedder>,
    pub(crate) history: Arc<HistoryStore>,
    pub(crate) builder: Mutex<MemoryBuilder>,
    pub(crate) retriever: HybridRetriever,
    pub(crate) config: Config,
    pub(crate) scope: Scope,
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
    pub fn builder() -> crate::MemeBuilder {
        crate::MemeBuilder::new()
    }

    /// Add a single dialogue turn.
    ///
    /// When the internal buffer reaches `window_size`, entries are automatically
    /// extracted, reconciled against existing memories, and stored.
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
            self.ingest_entries(&entries).await?;
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
            self.ingest_entries(&entries).await?;
        }
        Ok(())
    }

    /// Process any remaining dialogues in the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction or storage fails.
    pub async fn flush(&self) -> Result<()> {
        let mut builder = self.builder.lock().await;
        let entries = builder.finalize().await?;
        if !entries.is_empty() {
            self.ingest_entries(&entries).await?;
        }
        Ok(())
    }

    /// Add a raw text fact directly (bypasses dialogue windowing).
    ///
    /// The text is embedded, reconciled against existing memories,
    /// and stored as a single `MemoryEntry`.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or storage fails.
    #[tracing::instrument(skip(self))]
    pub async fn add(&self, content: &str) -> Result<()> {
        if content.is_empty() {
            return Err(Error::validation("content must not be empty"));
        }
        let entry = Memory::new(content);
        self.ingest_entries(&[entry]).await
    }

    /// Import pre-existing memory entries by recomputing embeddings and storing them.
    ///
    /// Skips reconciliation — entries are stored as-is (useful for migration/restore).
    ///
    /// # Errors
    ///
    /// Returns an error if embedding computation or storage fails.
    #[tracing::instrument(skip(self, entries), fields(count = entries.len()))]
    pub async fn import(&self, entries: &[Memory]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut scoped: Vec<Memory> = entries.to_vec();
        for entry in &mut scoped {
            if entry.user_id.is_none() {
                entry.user_id.clone_from(&self.scope.user_id);
            }
            if entry.session_id.is_none() {
                entry.session_id.clone_from(&self.scope.session_id);
            }
        }
        let texts: Vec<&str> = scoped.iter().map(|e| e.content.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;
        self.store.add_entries(&scoped, &vectors).await?;
        for entry in &scoped {
            if let Err(e) = self
                .history
                .record(
                    entry.id,
                    EventType::Add,
                    None,
                    Some(&entry.content),
                    &self.scope,
                )
                .await
            {
                tracing::warn!(memory_id = %entry.id, error = %e, "history record failed");
            }
        }
        Ok(())
    }

    /// Retrieve a single memory entry by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get(&self, id: Uuid) -> Result<Option<Memory>> {
        self.store.get_by_id(id).await
    }

    /// Update an existing memory entry's content.
    ///
    /// Re-embeds the new content, re-extracts structured metadata via LLM,
    /// and replaces the old entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the entry does not exist.
    pub async fn update(&self, id: Uuid, new_content: &str) -> Result<()> {
        let existing = self
            .store
            .get_by_id(id)
            .await?
            .ok_or_else(|| Error::NotFound { id: id.to_string() })?;

        let mut updated = existing.clone();
        updated.content = new_content.to_owned();
        self.re_extract_metadata(&mut updated).await;

        let vecs = self.embedder.encode_documents(&[new_content]).await?;
        let vec = vecs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty embedding".into()))?;
        self.store.update_entry(&updated, &vec).await?;
        if let Err(e) = self
            .history
            .record(
                id,
                EventType::Update,
                Some(&existing.content),
                Some(new_content),
                &self.scope,
            )
            .await
        {
            tracing::warn!(memory_id = %id, error = %e, "history record failed");
        }
        Ok(())
    }

    /// Re-extract structured metadata (keywords, persons, entities, etc.) from
    /// an entry's restatement via a lightweight LLM call.
    async fn re_extract_metadata(&self, entry: &mut Memory) {
        let prompt = llm::prompt::re_extract(&entry.content);
        let messages = vec![
            llm::Message::system("Extract structured metadata. Output valid JSON only."),
            llm::Message::user(prompt),
        ];
        let opts = llm::ChatOptions {
            temperature: 0.0,
            json_mode: true,
        };
        match self
            .llm
            .chat_structured::<ReExtractResponse>(&messages, &opts)
            .await
        {
            Ok(resp) => resp.apply_to(entry),
            Err(e) => {
                tracing::warn!(error = %e, "metadata re-extraction failed, keeping existing fields");
            }
        }
    }

    /// Delete a memory entry by ID.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] if the entry does not exist.
    pub async fn delete(&self, id: Uuid) -> Result<()> {
        let existing = self
            .store
            .get_by_id(id)
            .await?
            .ok_or_else(|| Error::NotFound { id: id.to_string() })?;

        self.store.delete_entries(&[id.to_string()]).await?;
        if let Err(e) = self
            .history
            .record(
                id,
                EventType::Delete,
                Some(&existing.content),
                None,
                &self.scope,
            )
            .await
        {
            tracing::warn!(memory_id = %id, error = %e, "history record failed");
        }
        Ok(())
    }

    /// Search memories by query text (semantic search).
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search(&self, query: &str) -> Result<Vec<Memory>> {
        let query_vec = self.embedder.encode_query(query).await?;
        self.store
            .semantic_search(&query_vec, self.config.pipeline.semantic_top_k, &self.scope)
            .await
    }

    /// Get the history of changes for a specific memory entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the history query fails.
    pub async fn history(&self, memory_id: Uuid) -> Result<Vec<Event>> {
        self.history.get_history(memory_id, &self.scope).await
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
        let answer =
            pipeline::generator::generate(&self.llm, question, &contexts, &self.config.pipeline)
                .await?;
        tracing::info!(contexts = contexts.len(), "answer generated");
        Ok(answer)
    }

    /// Get all stored memory entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn list(&self) -> Result<Vec<Memory>> {
        self.store.get_all(&self.scope).await
    }

    /// Execute the full hybrid retrieval pipeline (planning + multi-view search).
    ///
    /// Unlike [`search`](Self::search) which only does semantic ANN,
    /// this method uses intent-aware planning, keyword search, structured
    /// metadata filtering, and optional reflection.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<Memory>> {
        self.retriever.retrieve(query).await
    }

    /// Count stored memory entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn count(&self) -> Result<usize> {
        self.store.count(&self.scope).await
    }

    /// Clear stored memories for the current scope.
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

    async fn ingest_entries(&self, entries: &[Memory]) -> Result<()> {
        let mut scoped: Vec<Memory> = entries.to_vec();
        for entry in &mut scoped {
            if entry.user_id.is_none() {
                entry.user_id.clone_from(&self.scope.user_id);
            }
            if entry.session_id.is_none() {
                entry.session_id.clone_from(&self.scope.session_id);
            }
        }

        let texts: Vec<&str> = scoped.iter().map(|e| e.content.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;

        let existing_count = self.store.count(&self.scope).await?;
        if existing_count == 0 {
            self.store.add_entries(&scoped, &vectors).await?;
            self.record_history_batch(&scoped, EventType::Add).await;
            return Ok(());
        }

        let (to_add, vecs_add, deletes) =
            pipeline::reconciler::reconcile(&self.llm, &self.store, &self.scope, &scoped, &vectors)
                .await?;

        if !deletes.is_empty() {
            for (uid, old_content) in &deletes {
                if let Err(e) = self
                    .history
                    .record(
                        *uid,
                        EventType::Delete,
                        Some(old_content),
                        None,
                        &self.scope,
                    )
                    .await
                {
                    tracing::warn!(memory_id = %uid, error = %e, "history record failed");
                }
            }
            let ids: Vec<String> = deletes.iter().map(|(uid, _)| uid.to_string()).collect();
            self.store.delete_entries(&ids).await?;
            tracing::info!(count = deletes.len(), "deleted superseded memories");
        }

        if !to_add.is_empty() {
            self.store.add_entries(&to_add, &vecs_add).await?;
            self.record_history_batch(&to_add, EventType::Add).await;
        }
        Ok(())
    }

    async fn record_history_batch(&self, entries: &[Memory], event_type: EventType) {
        for entry in entries {
            if let Err(e) = self
                .history
                .record(
                    entry.id,
                    event_type,
                    None,
                    Some(&entry.content),
                    &self.scope,
                )
                .await
            {
                tracing::warn!(memory_id = %entry.id, error = %e, "history record failed");
            }
        }
    }
}
