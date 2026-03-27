//! [`Meme`] — the public-facing API for the memory system.
//!
//! Provides CRUD operations, dialogue ingestion, Q&A, and lifecycle
//! reconciliation behind a single ergonomic struct.

use std::sync::Arc;

use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::Config;
use crate::embedding::Embedder;
use crate::error::{Error, Result};
use crate::llm::{self, LlmClient, ReExtractResponse};
use crate::model::{Dialogue, Event, EventType, Memory};
use crate::pipeline::{self, Extractor, HybridRetriever};
use crate::store::{self, ConsolidationStats, HistoryStore, VectorStore};

/// The main entry point for the meme memory system.
///
/// Wraps the three-stage pipeline (compression, reconciliation, retrieval)
/// behind a simple async API with full CRUD and history tracking.
pub struct Meme {
    pub(crate) llm: Arc<LlmClient>,
    pub(crate) store: Arc<VectorStore>,
    pub(crate) embedder: Arc<Embedder>,
    pub(crate) history: Arc<HistoryStore>,
    pub(crate) extractor: Mutex<Extractor>,
    pub(crate) retriever: HybridRetriever,
    pub(crate) config: Config,
    pub(crate) namespace: Option<String>,
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

    /// Add dialogues to the memory system.
    ///
    /// Dialogues are buffered and processed through the LLM extraction pipeline
    /// in windows for optimal token efficiency. When the buffer reaches
    /// `window_size`, extraction is triggered automatically.
    ///
    /// Call [`flush`](Self::flush) after the last batch to ensure all
    /// buffered dialogues are processed.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction or storage fails.
    #[tracing::instrument(skip(self, dialogues), fields(count = dialogues.len()))]
    pub async fn add(&self, dialogues: &[Dialogue]) -> Result<()> {
        if dialogues.is_empty() {
            return Ok(());
        }
        let mut extractor = self.extractor.lock().await;
        let entries = extractor.add_dialogues(dialogues.to_vec()).await?;
        drop(extractor);
        if !entries.is_empty() {
            self.ingest_entries(&entries).await?;
        }
        Ok(())
    }

    /// Flush the dialogue buffer — process any remaining buffered dialogues.
    ///
    /// Must be called after the last [`add`](Self::add) to ensure no dialogues
    /// are left unprocessed in the internal buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction or storage fails.
    pub async fn flush(&self) -> Result<()> {
        let mut extractor = self.extractor.lock().await;
        let entries = extractor.flush().await?;
        drop(extractor);
        if !entries.is_empty() {
            self.ingest_entries(&entries).await?;
        }
        Ok(())
    }

    /// Store a fact directly into memory (bypasses dialogue windowing).
    ///
    /// The text is embedded, reconciled against existing memories,
    /// and stored as a single [`Memory`].
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or storage fails.
    #[tracing::instrument(skip(self))]
    pub async fn put(&self, content: &str) -> Result<()> {
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
        self.apply_namespace(&mut scoped);
        let texts: Vec<&str> = scoped.iter().map(|e| e.content.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;
        self.store.add_entries(&scoped, &vectors).await?;
        for entry in &scoped {
            self.record_event(entry.id, EventType::Add, None, Some(&entry.content))
                .await;
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
        updated.updated_at = Some(chrono::Utc::now());
        self.re_extract_metadata(&mut updated).await;

        let vecs = self.embedder.encode_documents(&[new_content]).await?;
        let vec = vecs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("empty embedding".into()))?;
        self.store.update_entry(&updated, &vec).await?;
        self.record_event(
            id,
            EventType::Update,
            Some(&existing.content),
            Some(new_content),
        )
        .await;
        Ok(())
    }

    /// Re-extract structured metadata (keywords, persons, entities, etc.) from
    /// a memory's content via a lightweight LLM call.
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

        self.store.delete_entries(&[id]).await?;
        self.record_event(id, EventType::Delete, Some(&existing.content), None)
            .await;
        Ok(())
    }

    /// Search memories using the full hybrid retrieval pipeline.
    ///
    /// Combines semantic ANN search, keyword (FTS/LIKE) search, and structured
    /// metadata filtering with optional LLM-driven query planning and reflection.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search(&self, query: &str) -> Result<Vec<Memory>> {
        self.retriever.retrieve(query).await
    }

    /// Get the history of changes for a specific memory entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the history query fails.
    pub async fn history(&self, memory_id: Uuid) -> Result<Vec<Event>> {
        self.history.get_history(memory_id, self.ns()).await
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
            pipeline::generate(&self.llm, question, &contexts, &self.config.pipeline).await?;
        tracing::info!(contexts = contexts.len(), "answer generated");
        Ok(answer)
    }

    /// Get all stored memory entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn list(&self) -> Result<Vec<Memory>> {
        self.store.get_all(self.ns()).await
    }

    /// Count stored memory entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn count(&self) -> Result<usize> {
        self.store.count(self.ns()).await
    }

    /// Clear stored memories for the current scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the clear operation fails.
    pub async fn clear(&self) -> Result<()> {
        self.store.clear(self.ns()).await
    }

    /// Consolidate memory: decay old entries, merge near-duplicates, prune low-importance.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or deleting entries fails.
    pub async fn consolidate(
        &self,
        max_age_days: u32,
        decay_factor: f64,
        merge_threshold: f64,
        min_importance: f64,
    ) -> Result<ConsolidationStats> {
        store::consolidate(
            &self.store,
            max_age_days,
            decay_factor,
            merge_threshold,
            min_importance,
            self.ns(),
        )
        .await
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    fn ns(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    fn apply_namespace(&self, entries: &mut [Memory]) {
        for entry in entries {
            if entry.namespace.is_none() {
                entry.namespace.clone_from(&self.namespace);
            }
        }
    }

    async fn record_event(
        &self,
        memory_id: Uuid,
        event_type: EventType,
        old: Option<&str>,
        new: Option<&str>,
    ) {
        if let Err(e) = self
            .history
            .record(memory_id, event_type, old, new, self.ns())
            .await
        {
            tracing::warn!(%memory_id, error = %e, "history record failed");
        }
    }

    async fn ingest_entries(&self, entries: &[Memory]) -> Result<()> {
        let mut scoped: Vec<Memory> = entries.to_vec();
        self.apply_namespace(&mut scoped);

        let texts: Vec<&str> = scoped.iter().map(|e| e.content.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;

        let existing_count = self.store.count(self.ns()).await?;
        if existing_count == 0 {
            self.store.add_entries(&scoped, &vectors).await?;
            for entry in &scoped {
                self.record_event(entry.id, EventType::Add, None, Some(&entry.content))
                    .await;
            }
            return Ok(());
        }

        let (to_add, vecs_add, deletes) =
            pipeline::reconcile(&self.llm, &self.store, self.ns(), &scoped, &vectors).await?;

        if !deletes.is_empty() {
            for (uid, old_content) in &deletes {
                self.record_event(*uid, EventType::Delete, Some(old_content), None)
                    .await;
            }
            let ids: Vec<Uuid> = deletes.iter().map(|(uid, _)| *uid).collect();
            self.store.delete_entries(&ids).await?;
            tracing::info!(count = deletes.len(), "deleted superseded memories");
        }

        if !to_add.is_empty() {
            self.store.add_entries(&to_add, &vecs_add).await?;
            for entry in &to_add {
                self.record_event(entry.id, EventType::Add, None, Some(&entry.content))
                    .await;
            }
        }
        Ok(())
    }
}
