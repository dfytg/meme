//! Cross-session types — session lifecycle, events, observations, and consolidation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::MemoryEntry;

/// Lifecycle status for a memory session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    /// Session is currently active.
    Active,
    /// Session completed successfully.
    Completed,
    /// Session ended with an error.
    Failed,
}

/// Kinds of events captured during a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A chat message.
    Message,
    /// A tool invocation.
    ToolUse,
    /// A file modification.
    FileChange,
    /// A free-form note.
    Note,
    /// A system-generated event.
    System,
}

/// Semantic observation categories extracted from sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObservationType {
    /// An architectural or design decision.
    Decision,
    /// A bug fix.
    Bugfix,
    /// A new feature.
    Feature,
    /// A code refactor.
    Refactor,
    /// A discovery or learning.
    Discovery,
    /// A generic change.
    Change,
}

/// Redaction levels for event payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RedactionLevel {
    /// No redaction.
    #[default]
    None,
    /// Partial redaction (sensitive values masked).
    Partial,
    /// Full redaction (payload removed).
    Full,
}

/// A conversation session record persisted in SQLite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Database row ID.
    pub row_id: Option<i64>,
    /// Tenant identifier for multi-tenant isolation.
    pub tenant_id: String,
    /// External session identifier from the content platform.
    pub content_session_id: String,
    /// Internal memory session identifier.
    pub memory_session_id: Uuid,
    /// Project name or path.
    pub project: String,
    /// The user's initial prompt/request.
    pub user_prompt: Option<String>,
    /// When the session started.
    pub started_at: DateTime<Utc>,
    /// When the session ended.
    pub ended_at: Option<DateTime<Utc>>,
    /// Current lifecycle status.
    pub status: SessionStatus,
    /// Arbitrary JSON metadata.
    pub metadata_json: Option<String>,
}

/// A single event during a session timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Database row ID.
    pub event_id: Option<i64>,
    /// The session this event belongs to.
    pub memory_session_id: Uuid,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Event kind.
    pub kind: EventKind,
    /// Short title.
    pub title: Option<String>,
    /// JSON payload.
    pub payload_json: Option<String>,
    /// Redaction level applied.
    pub redaction_level: RedactionLevel,
}

/// An observation extracted from a session for cross-session memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossObservation {
    /// Database row ID.
    pub obs_id: Option<i64>,
    /// The session this observation was extracted from.
    pub memory_session_id: Uuid,
    /// When the observation was recorded.
    pub timestamp: DateTime<Utc>,
    /// Observation category.
    pub obs_type: ObservationType,
    /// Short title.
    pub title: String,
    /// Subtitle or brief detail.
    pub subtitle: Option<String>,
    /// JSON-encoded list of facts.
    pub facts_json: Option<String>,
    /// Narrative description.
    pub narrative: Option<String>,
    /// JSON-encoded list of concepts.
    pub concepts_json: Option<String>,
    /// JSON-encoded list of affected files.
    pub files_json: Option<String>,
    /// Reference to the vector store entry.
    pub vector_ref: Option<String>,
}

/// Summary generated when a session ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Database row ID.
    pub summary_id: Option<i64>,
    /// The session this summary belongs to.
    pub memory_session_id: Uuid,
    /// When the summary was generated.
    pub timestamp: DateTime<Utc>,
    /// What was requested.
    pub request: Option<String>,
    /// What was investigated.
    pub investigated: Option<String>,
    /// What was learned.
    pub learned: Option<String>,
    /// What was completed.
    pub completed: Option<String>,
    /// Suggested next steps.
    pub next_steps: Option<String>,
    /// Reference to the vector store entry.
    pub vector_ref: Option<String>,
}

/// Traceability mapping from vectors back to source evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    /// Database row ID.
    pub link_id: Option<i64>,
    /// The memory entry this link points to.
    pub memory_entry_id: Uuid,
    /// Kind of source (`"observation"`, `"summary"`, `"event"`).
    pub source_kind: String,
    /// Source row ID.
    pub source_id: i64,
    /// Relevance score.
    pub score: f64,
    /// When the link was created.
    pub timestamp: DateTime<Utc>,
}

/// Memory entry with cross-session provenance fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEntry {
    /// The base memory entry.
    #[serde(flatten)]
    pub entry: MemoryEntry,
    /// Tenant identifier.
    pub tenant_id: String,
    /// Session that produced this entry.
    pub memory_session_id: Uuid,
    /// Kind of source evidence.
    pub source_kind: String,
    /// Source row ID.
    pub source_id: Option<i64>,
    /// Importance score (0.0..=1.0).
    pub importance: f64,
    /// When this entry became valid.
    pub valid_from: Option<DateTime<Utc>>,
    /// When this entry expires.
    pub valid_to: Option<DateTime<Utc>>,
    /// If superseded, the ID of the replacement entry.
    pub superseded_by: Option<Uuid>,
}

/// Payload injected at session start with relevant cross-session context.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextBundle {
    /// Recent session summaries.
    pub session_summaries: Vec<SessionSummary>,
    /// Timeline observations.
    pub timeline_observations: Vec<CrossObservation>,
    /// Relevant memory entries.
    pub memory_entries: Vec<CrossEntry>,
    /// Estimated total token count.
    pub total_tokens_estimate: usize,
}

impl ContextBundle {
    /// Render the bundle into a string capped by a token estimate.
    #[must_use]
    pub fn render(&self, max_tokens: usize) -> String {
        let estimate_tokens = |text: &str| text.split_whitespace().count();

        let mut lines: Vec<String> = Vec::new();
        let mut token_count = 0usize;

        let mut try_add = |line: String| {
            let next = estimate_tokens(&line);
            if token_count + next <= max_tokens {
                token_count += next;
                lines.push(line);
            }
        };

        if !self.session_summaries.is_empty() {
            try_add("Session summaries:".to_owned());
            for s in &self.session_summaries {
                let text = s
                    .completed
                    .as_deref()
                    .or(s.learned.as_deref())
                    .or(s.investigated.as_deref())
                    .or(s.request.as_deref())
                    .unwrap_or("Summary available.");
                try_add(format!("- {text}"));
            }
        }

        if !self.timeline_observations.is_empty() {
            try_add("Timeline observations:".to_owned());
            for obs in &self.timeline_observations {
                let detail = obs
                    .subtitle
                    .as_deref()
                    .or(obs.narrative.as_deref())
                    .unwrap_or("");
                if detail.is_empty() {
                    try_add(format!("- {}", obs.title));
                } else {
                    try_add(format!("- {}: {detail}", obs.title));
                }
            }
        }

        if !self.memory_entries.is_empty() {
            try_add("Memory entries:".to_owned());
            for e in &self.memory_entries {
                try_add(format!("- {}", e.entry.restatement));
            }
        }

        lines.join("\n")
    }
}

/// Report returned when a session finishes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FinalizationReport {
    /// Session that was finalized.
    pub memory_session_id: Uuid,
    /// Number of observations extracted.
    pub observations_count: usize,
    /// Whether a summary was generated.
    pub summary_generated: bool,
    /// Number of memory entries stored.
    pub entries_stored: usize,
    /// Whether consolidation was triggered.
    pub consolidation_triggered: bool,
}

/// Record of a consolidation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationRun {
    /// Database row ID.
    pub run_id: Option<i64>,
    /// Tenant identifier.
    pub tenant_id: String,
    /// When the run occurred.
    pub timestamp: DateTime<Utc>,
    /// JSON-encoded policy used.
    pub policy_json: Option<String>,
    /// JSON-encoded statistics.
    pub stats_json: Option<String>,
}
