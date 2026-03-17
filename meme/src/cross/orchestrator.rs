//! Cross-session orchestrator — top-level facade for the cross-session memory system.

use std::sync::Arc;

use uuid::Uuid;

use super::collector::EventCollector;
use super::consolidation::{ConsolidationPolicy, ConsolidationStats, ConsolidationWorker};
use super::extractor::ObservationExtractor;
use super::injector::ContextInjector;
use crate::config::{Config, CrossConfig};
use crate::embedding::Embedder;
use crate::error::Result;
use crate::model::{ContextBundle, CrossEntry, FinalizationReport, Session, SessionSummary};
use crate::store::{SqliteStore, VectorStore};

/// Result of starting a new session.
#[derive(Debug)]
pub struct StartSessionResult {
    /// The memory session ID assigned to this session.
    pub memory_session_id: Uuid,
    /// Context bundle injected from previous sessions.
    pub context: ContextBundle,
    /// Rendered context string for the agent's system prompt.
    pub context_text: String,
}

/// Top-level entry point for the cross-session memory system.
///
/// Wires together `SQLite` storage, session management, context injection,
/// event collection, and consolidation into a single facade.
pub struct CrossOrchestrator {
    db: SqliteStore,
    vector_store: Option<Arc<VectorStore>>,
    embedder: Option<Arc<Embedder>>,
    project: String,
    tenant_id: String,
    cross_cfg: CrossConfig,
}

impl std::fmt::Debug for CrossOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossOrchestrator")
            .field("project", &self.project)
            .field("tenant_id", &self.tenant_id)
            .finish()
    }
}

#[allow(clippy::future_not_send)]
impl CrossOrchestrator {
    /// Create a new orchestrator without vector store (SQLite-only mode).
    ///
    /// Consolidation and semantic context injection will be unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the `SQLite` database cannot be opened.
    pub fn new(project: &str, config: &Config) -> Result<Self> {
        let db = SqliteStore::open(&config.cross.db_path)?;
        Ok(Self {
            db,
            vector_store: None,
            embedder: None,
            project: project.to_owned(),
            tenant_id: "default".to_owned(),
            cross_cfg: config.cross.clone(),
        })
    }

    /// Create with vector store and embedding provider for full functionality.
    ///
    /// Enables consolidation and semantic context injection.
    ///
    /// # Errors
    ///
    /// Returns an error if the `SQLite` database cannot be opened.
    pub fn with_stores(
        project: &str,
        config: &Config,
        vector_store: Arc<VectorStore>,
        embedder: Arc<Embedder>,
    ) -> Result<Self> {
        let db = SqliteStore::open(&config.cross.db_path)?;
        Ok(Self {
            db,
            vector_store: Some(vector_store),
            embedder: Some(embedder),
            project: project.to_owned(),
            tenant_id: "default".to_owned(),
            cross_cfg: config.cross.clone(),
        })
    }

    /// Create with a custom tenant ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the `SQLite` database cannot be opened.
    pub fn with_tenant(project: &str, tenant_id: &str, config: &Config) -> Result<Self> {
        let db = SqliteStore::open(&config.cross.db_path)?;
        Ok(Self {
            db,
            vector_store: None,
            embedder: None,
            project: project.to_owned(),
            tenant_id: tenant_id.to_owned(),
            cross_cfg: config.cross.clone(),
        })
    }

    /// Start a new session with automatic context injection from previous sessions.
    ///
    /// When a vector store and embedding provider are available, Tier 3 semantic
    /// search is performed against `user_prompt` for richer context injection.
    ///
    /// # Errors
    ///
    /// Returns an error if session creation or context retrieval fails.
    pub async fn start_session(
        &self,
        content_session_id: &str,
        user_prompt: Option<&str>,
    ) -> Result<StartSessionResult> {
        let session = Session {
            row_id: None,
            tenant_id: self.tenant_id.clone(),
            content_session_id: content_session_id.to_owned(),
            memory_session_id: Uuid::new_v4(),
            project: self.project.clone(),
            user_prompt: user_prompt.map(String::from),
            started_at: chrono::Utc::now(),
            ended_at: None,
            status: crate::model::SessionStatus::Active,
            metadata_json: None,
        };
        let row_id = self.db.insert_session(&session)?;
        let session = Session {
            row_id: Some(row_id),
            ..session
        };

        let injector = ContextInjector::new(&self.db, self.cross_cfg.max_context_tokens);
        let context = injector
            .build_context(
                &self.project,
                user_prompt,
                self.vector_store.as_deref(),
                self.embedder.as_deref(),
            )
            .await?;
        let context_text = context.render(self.cross_cfg.max_context_tokens);

        Ok(StartSessionResult {
            memory_session_id: session.memory_session_id,
            context,
            context_text,
        })
    }

    /// Record a chat message event in the current session.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be recorded.
    pub fn record_message(
        &self,
        memory_session_id: &Uuid,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let collector = EventCollector::new(&self.db);
        collector.record_message(memory_session_id, role, content)?;
        Ok(())
    }

    /// Record a tool use event in the current session.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be recorded.
    pub fn record_tool_use(
        &self,
        memory_session_id: &Uuid,
        tool_name: &str,
        tool_input: &str,
        tool_output: &str,
    ) -> Result<()> {
        let collector = EventCollector::new(&self.db);
        collector.record_tool_use(memory_session_id, tool_name, tool_input, tool_output)?;
        Ok(())
    }

    /// Stop a session, generate summary, and store observations.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    pub fn stop_session(&self, memory_session_id: &Uuid) -> Result<FinalizationReport> {
        self.db.update_session_status(
            memory_session_id,
            crate::model::SessionStatus::Completed,
            Some(chrono::Utc::now()),
        )?;

        let events = self.db.get_events(memory_session_id)?;
        let event_count = events.len();

        let observations = ObservationExtractor::extract_from_events(&events, memory_session_id);
        let observations_count = observations.len();
        for obs in &observations {
            let _ = self.db.insert_observation(obs);
        }

        let session = self.db.get_session(memory_session_id)?;
        let value = ObservationExtractor::estimate_session_value(&events);

        let summary = SessionSummary {
            summary_id: None,
            memory_session_id: *memory_session_id,
            timestamp: chrono::Utc::now(),
            request: session.and_then(|s| s.user_prompt),
            investigated: None,
            learned: if observations_count > 0 {
                Some(format!(
                    "Extracted {observations_count} observations from {event_count} events."
                ))
            } else {
                None
            },
            completed: Some(format!(
                "Session completed with {event_count} events (value: {value:.2})."
            )),
            next_steps: None,
            vector_ref: None,
        };
        self.db.upsert_summary(&summary)?;

        let report = FinalizationReport {
            memory_session_id: *memory_session_id,
            observations_count,
            summary_generated: true,
            entries_stored: 0,
            consolidation_triggered: false,
        };

        tracing::info!(
            session_id = %memory_session_id,
            events = event_count,
            observations = observations_count,
            value = value,
            "session finalized"
        );

        Ok(report)
    }

    /// Trigger memory consolidation: decay old entries, merge near-duplicates,
    /// and prune low-importance entries.
    ///
    /// Requires a vector store to be configured (via [`Self::with_stores`]).
    /// Falls back to a no-op record if no vector store is available.
    ///
    /// # Errors
    ///
    /// Returns an error if consolidation or storage operations fail.
    pub async fn consolidate(&self) -> Result<ConsolidationStats> {
        let policy = ConsolidationPolicy::from_config(&self.cross_cfg);
        let worker = ConsolidationWorker::new(policy);

        let Some(store) = &self.vector_store else {
            tracing::warn!("consolidation skipped: no vector store configured");
            let stats = ConsolidationStats::default();
            ConsolidationWorker::record_run(&self.db, &self.tenant_id, &policy, &stats)?;
            return Ok(stats);
        };

        let pairs = store.get_all_with_vectors().await?;
        if pairs.is_empty() {
            let stats = ConsolidationStats::default();
            ConsolidationWorker::record_run(&self.db, &self.tenant_id, &policy, &stats)?;
            return Ok(stats);
        }

        let limit = policy.max_entries_per_run;
        let mut cross_entries: Vec<CrossEntry> = pairs
            .iter()
            .take(limit)
            .map(|(entry, _)| CrossEntry {
                entry: entry.clone(),
                tenant_id: self.tenant_id.clone(),
                memory_session_id: Uuid::new_v4(),
                source_kind: "vector_store".to_owned(),
                source_id: None,
                importance: 1.0,
                valid_from: entry.timestamp,
                valid_to: None,
                superseded_by: None,
            })
            .collect();

        let vectors: Vec<Vec<f32>> = pairs.iter().take(limit).map(|(_, v)| v.clone()).collect();

        let actions = worker.compute(&mut cross_entries, &vectors);

        // Apply deletions: remove pruned + superseded entries from the vector store.
        let mut ids_to_delete: Vec<String> = actions.pruned.clone();
        for (loser_id, _) in &actions.superseded {
            ids_to_delete.push(loser_id.clone());
        }
        if !ids_to_delete.is_empty() {
            let deleted = store.delete_entries(&ids_to_delete).await?;
            tracing::info!(deleted, "consolidation deletions applied");
        }

        ConsolidationWorker::record_run(&self.db, &self.tenant_id, &policy, &actions.stats)?;
        Ok(actions.stats)
    }

    /// List recent sessions.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_sessions(&self, limit: usize) -> Result<Vec<Session>> {
        self.db.list_sessions(&self.project, limit)
    }
}
