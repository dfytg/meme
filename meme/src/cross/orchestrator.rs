//! Cross-session orchestrator — top-level facade for the cross-session memory system.

use std::sync::Arc;

use uuid::Uuid;

use super::collector::EventCollector;
use super::consolidation::{ConsolidationPolicy, ConsolidationStats, ConsolidationWorker};
use super::injector::ContextInjector;
use super::session::SessionManager;
use crate::config::{Config, CrossConfig};
use crate::error::Result;
use crate::model::{ContextBundle, FinalizationReport, Session, SessionSummary};
use crate::store::SqliteStore;

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
/// Wires together SQLite storage, session management, context injection,
/// event collection, and consolidation into a single facade.
pub struct CrossOrchestrator {
    db: SqliteStore,
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

impl CrossOrchestrator {
    /// Create a new orchestrator.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQLite database cannot be opened.
    pub fn new(project: &str, config: &Config) -> Result<Self> {
        let db = SqliteStore::open(&config.cross.db_path)?;
        Ok(Self {
            db,
            project: project.to_owned(),
            tenant_id: "default".to_owned(),
            cross_cfg: config.cross.clone(),
        })
    }

    /// Create with a custom tenant ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQLite database cannot be opened.
    pub fn with_tenant(project: &str, tenant_id: &str, config: &Config) -> Result<Self> {
        let db = SqliteStore::open(&config.cross.db_path)?;
        Ok(Self {
            db,
            project: project.to_owned(),
            tenant_id: tenant_id.to_owned(),
            cross_cfg: config.cross.clone(),
        })
    }

    /// Start a new session with automatic context injection from previous sessions.
    ///
    /// # Errors
    ///
    /// Returns an error if session creation or context retrieval fails.
    pub fn start_session(
        &self,
        content_session_id: &str,
        user_prompt: Option<&str>,
    ) -> Result<StartSessionResult> {
        let mgr = SessionManager::new(&self.db);
        let session = mgr.start(
            content_session_id,
            &self.project,
            user_prompt,
            &self.tenant_id,
        )?;

        let injector = ContextInjector::new(&self.db, self.cross_cfg.max_context_tokens);
        let context = injector.build_context(&self.project)?;
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
    pub fn record_message(&self, memory_session_id: &Uuid, content: &str) -> Result<()> {
        let collector = EventCollector::new(&self.db, crate::model::RedactionLevel::None);
        collector.record_message(memory_session_id, content)?;
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
        let collector = EventCollector::new(&self.db, crate::model::RedactionLevel::None);
        collector.record_tool_use(memory_session_id, tool_name, tool_input, tool_output)?;
        Ok(())
    }

    /// Stop a session, generate summary, and store observations.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    pub fn stop_session(&self, memory_session_id: &Uuid) -> Result<FinalizationReport> {
        let mgr = SessionManager::new(&self.db);
        mgr.stop(memory_session_id)?;

        // Generate a basic summary from events.
        let events = self.db.get_events(memory_session_id)?;
        let event_count = events.len();

        let summary = SessionSummary {
            summary_id: None,
            memory_session_id: *memory_session_id,
            timestamp: chrono::Utc::now(),
            request: self
                .db
                .get_session(memory_session_id)?
                .and_then(|s| s.user_prompt),
            investigated: None,
            learned: None,
            completed: Some(format!(
                "Session completed with {event_count} events recorded."
            )),
            next_steps: None,
            vector_ref: None,
        };
        self.db.upsert_summary(&summary)?;

        let report = FinalizationReport {
            memory_session_id: *memory_session_id,
            observations_count: 0,
            summary_generated: true,
            entries_stored: 0,
            consolidation_triggered: false,
        };

        tracing::info!(
            session_id = %memory_session_id,
            events = event_count,
            "session finalized"
        );

        Ok(report)
    }

    /// End a session (final cleanup after stop).
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be found.
    pub fn end_session(&self, memory_session_id: &Uuid) -> Result<()> {
        tracing::info!(session_id = %memory_session_id, "session ended");
        Ok(())
    }

    /// Manually trigger memory consolidation.
    ///
    /// # Errors
    ///
    /// Returns an error if consolidation fails.
    pub fn consolidate(&self) -> Result<ConsolidationStats> {
        let policy = ConsolidationPolicy::from_config(&self.cross_cfg);
        let worker = ConsolidationWorker::new(&self.db, policy, &self.tenant_id);
        worker.run()
    }

    /// List recent sessions.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_sessions(&self, limit: usize) -> Result<Vec<Session>> {
        let mgr = SessionManager::new(&self.db);
        mgr.list(&self.project, limit)
    }
}
