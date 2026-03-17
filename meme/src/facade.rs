//! [`Meme`] — the public-facing API for the memory system.
//!
//! Provides CRUD operations, dialogue ingestion, Q&A, and lifecycle
//! reconciliation behind a single ergonomic struct.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::future;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::Config;
use crate::embedding::Embedder;
use crate::error::{Error, Result};
use crate::llm::{self, LlmClient};
use crate::model::{Dialogue, EventType, MemoryEntry, MemoryEvent};
use crate::pipeline::{self, HybridRetriever, MemoryBuilder};
use crate::store::{HistoryStore, Scope, VectorStore};

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
    pub async fn finalize(&self) -> Result<()> {
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
        let entry = MemoryEntry::new(content);
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
        let texts: Vec<&str> = entries.iter().map(|e| e.restatement.as_str()).collect();
        let vectors = self.embedder.encode_documents(&texts).await?;
        self.store.add_entries(entries, &vectors).await?;
        for entry in entries.iter() {
            if let Err(e) = self
                .history
                .record(entry.id, EventType::Add, None, Some(&entry.restatement))
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
    pub async fn get(&self, id: Uuid) -> Result<Option<MemoryEntry>> {
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
        updated.restatement = new_content.to_owned();
        self.re_extract_metadata(&mut updated).await;

        let vec = self.embedder.encode_query(new_content).await?;
        self.store.update_entry(&updated, &vec).await?;
        if let Err(e) = self
            .history
            .record(
                id,
                EventType::Update,
                Some(&existing.restatement),
                Some(new_content),
            )
            .await
        {
            tracing::warn!(memory_id = %id, error = %e, "history record failed");
        }
        Ok(())
    }

    /// Re-extract structured metadata (keywords, persons, entities, etc.) from
    /// an entry's restatement via a lightweight LLM call.
    async fn re_extract_metadata(&self, entry: &mut MemoryEntry) {
        let prompt = llm::prompt::re_extract(&entry.restatement);
        let messages = vec![
            llm::Message::system("Extract structured metadata. Output valid JSON only."),
            llm::Message::user(prompt),
        ];
        let opts = llm::ChatOptions {
            temperature: 0.0,
            json_mode: true,
        };
        let Ok(response) = self.llm.chat(&messages, &opts).await else {
            tracing::warn!("metadata re-extraction LLM call failed, keeping existing fields");
            return;
        };
        let Ok(data) = llm::extract_json_from_text(&response) else {
            tracing::warn!("metadata re-extraction parse failed, keeping existing fields");
            return;
        };

        if let Some(kw) = data["keywords"].as_array() {
            entry.keywords = kw
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
        if let Some(ps) = data["persons"].as_array() {
            entry.persons = ps
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
        if let Some(es) = data["entities"].as_array() {
            entry.entities = es
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
        entry.location = data["location"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .map(String::from);
        entry.topic = data["topic"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .map(String::from);
        if let Some(ts) = data["timestamp"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            && let Ok(dt) = DateTime::parse_from_rfc3339(ts)
        {
            entry.timestamp = Some(dt.with_timezone(&Utc));
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
            .record(id, EventType::Delete, Some(&existing.restatement), None)
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
    pub async fn search(&self, query: &str) -> Result<Vec<MemoryEntry>> {
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
    pub async fn history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        self.history.get_history(memory_id).await
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
    pub async fn get_all(&self) -> Result<Vec<MemoryEntry>> {
        self.store.get_all(&self.scope).await
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

    async fn ingest_entries(&self, entries: &[MemoryEntry]) -> Result<()> {
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

        let existing_count = self.store.count(&self.scope).await?;
        if existing_count == 0 {
            self.store.add_entries(&scoped, &vectors).await?;
            self.record_history_batch(&scoped, EventType::Add).await;
            return Ok(());
        }

        let (to_add, vecs_add, ids_to_delete) = self.reconcile_memories(&scoped, &vectors).await?;

        if !ids_to_delete.is_empty() {
            self.record_history_deletes(&ids_to_delete).await?;
            self.store.delete_entries(&ids_to_delete).await?;
            tracing::info!(count = ids_to_delete.len(), "deleted superseded memories");
        }

        if !to_add.is_empty() {
            self.store.add_entries(&to_add, &vecs_add).await?;
            self.record_history_batch(&to_add, EventType::Add).await;
        }
        Ok(())
    }

    async fn record_history_batch(&self, entries: &[MemoryEntry], event_type: EventType) {
        for entry in entries {
            if let Err(e) = self
                .history
                .record(entry.id, event_type, None, Some(&entry.restatement))
                .await
            {
                tracing::warn!(memory_id = %entry.id, error = %e, "history record failed");
            }
        }
    }

    async fn record_history_deletes(&self, ids: &[String]) -> Result<()> {
        for id_str in ids {
            if let Ok(uid) = Uuid::parse_str(id_str) {
                let old = self.store.get_by_id(uid).await?.map(|e| e.restatement);
                if let Err(e) = self
                    .history
                    .record(uid, EventType::Delete, old.as_deref(), None)
                    .await
                {
                    tracing::warn!(memory_id = %uid, error = %e, "history record failed");
                }
            }
        }
        Ok(())
    }

    /// Reconcile new entries against existing memories using the LLM.
    ///
    /// Returns (entries to insert, their vectors, IDs to delete from store).
    async fn reconcile_memories(
        &self,
        entries: &[MemoryEntry],
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<MemoryEntry>, Vec<Vec<f32>>, Vec<String>)> {
        let similarity_top_k = 5;

        let new_facts: Vec<&str> = entries.iter().map(|e| e.restatement.as_str()).collect();

        let ann_futures: Vec<_> = vectors
            .iter()
            .map(|vec_i| {
                self.store
                    .semantic_search(vec_i, similarity_top_k, &self.scope)
            })
            .collect();
        let all_existing: Vec<Vec<MemoryEntry>> = future::try_join_all(ann_futures).await?;

        let mut existing_map: HashMap<Uuid, (usize, String)> = HashMap::new();
        for group in &all_existing {
            for entry in group {
                let next_idx = existing_map.len();
                existing_map
                    .entry(entry.id)
                    .or_insert_with(|| (next_idx, entry.restatement.clone()));
            }
        }

        if existing_map.is_empty() {
            return Ok((entries.to_vec(), vectors.to_vec(), Vec::new()));
        }

        let existing_indexed: Vec<(usize, &str)> = {
            let mut v: Vec<(usize, &str)> = existing_map
                .values()
                .map(|(idx, text)| (*idx, text.as_str()))
                .collect();
            v.sort_by_key(|(idx, _)| *idx);
            v
        };

        let idx_to_uuid: HashMap<usize, Uuid> = existing_map
            .iter()
            .map(|(uid, (idx, _))| (*idx, *uid))
            .collect();

        let prompt_text = llm::prompt::reconcile(&new_facts, &existing_indexed);
        let messages = vec![
            llm::Message::system(
                "You are a smart memory manager. You must output valid JSON format.",
            ),
            llm::Message::user(prompt_text),
        ];
        let opts = llm::ChatOptions {
            temperature: 0.1,
            json_mode: true,
        };

        let response = self.llm.chat(&messages, &opts).await?;
        let data = llm::extract_json_from_text(&response)?;

        let actions = data["actions"]
            .as_array()
            .or_else(|| {
                data.as_object()
                    .and_then(|obj| obj.values().find_map(|v| v.as_array()))
            })
            .cloned()
            .unwrap_or_default();

        let mut accepted = Vec::new();
        let mut accepted_vecs = Vec::new();
        let mut ids_to_delete = Vec::new();

        for action_val in &actions {
            let Some(new_idx) = parse_index(&action_val["new_index"]) else {
                continue;
            };
            let act = action_val["action"].as_str().unwrap_or("add");
            let existing_idx = parse_index(&action_val["existing_index"]);

            if new_idx >= entries.len() {
                continue;
            }

            let target_uid = existing_idx.and_then(|eidx| idx_to_uuid.get(&eidx));

            match act {
                "update" => {
                    if let Some(uid) = target_uid {
                        ids_to_delete.push(uid.to_string());
                    }
                    accepted.push(entries[new_idx].clone());
                    accepted_vecs.push(vectors[new_idx].clone());
                }
                "delete" => {
                    if let Some(uid) = target_uid {
                        ids_to_delete.push(uid.to_string());
                    }
                }
                "noop" | "duplicate" => {
                    tracing::debug!(
                        new = entries[new_idx].restatement.as_str(),
                        "skipping duplicate/noop"
                    );
                }
                _ => {
                    accepted.push(entries[new_idx].clone());
                    accepted_vecs.push(vectors[new_idx].clone());
                }
            }
        }

        let handled: HashSet<usize> = actions
            .iter()
            .filter_map(|a| parse_index(&a["new_index"]))
            .collect();
        for (i, entry) in entries.iter().enumerate() {
            if !handled.contains(&i) {
                accepted.push(entry.clone());
                accepted_vecs.push(vectors[i].clone());
            }
        }

        ids_to_delete.sort();
        ids_to_delete.dedup();

        Ok((accepted, accepted_vecs, ids_to_delete))
    }
}

/// Parse a JSON value as a `usize` index, tolerating both integers and string digits.
fn parse_index(val: &serde_json::Value) -> Option<usize> {
    val.as_u64()
        .map(|v| v as usize)
        .or_else(|| val.as_str().and_then(|s| s.parse::<usize>().ok()))
}
