//! Event collector — captures and optionally redacts session events.

use chrono::Utc;
use uuid::Uuid;

use crate::error::Result;
use crate::model::{EventKind, RedactionLevel, SessionEvent};
use crate::store::SqliteStore;

/// Collects events during a session with configurable redaction.
pub struct EventCollector<'a> {
    db: &'a SqliteStore,
    default_redaction: RedactionLevel,
}

impl std::fmt::Debug for EventCollector<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventCollector")
            .field("default_redaction", &self.default_redaction)
            .finish()
    }
}

impl<'a> EventCollector<'a> {
    /// Create a new event collector.
    pub fn new(db: &'a SqliteStore, default_redaction: RedactionLevel) -> Self {
        Self {
            db,
            default_redaction,
        }
    }

    /// Record a chat message event.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_message(&self, memory_session_id: &Uuid, content: &str) -> Result<i64> {
        self.record(memory_session_id, EventKind::Message, Some(content), None)
    }

    /// Record a tool use event.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_tool_use(
        &self,
        memory_session_id: &Uuid,
        tool_name: &str,
        tool_input: &str,
        tool_output: &str,
    ) -> Result<i64> {
        let payload = serde_json::json!({
            "tool_name": tool_name,
            "input": self.maybe_redact(tool_input),
            "output": self.maybe_redact(tool_output),
        });
        self.record(
            memory_session_id,
            EventKind::ToolUse,
            Some(tool_name),
            Some(&payload.to_string()),
        )
    }

    /// Record a file change event.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_file_change(
        &self,
        memory_session_id: &Uuid,
        file_path: &str,
        change_type: &str,
    ) -> Result<i64> {
        let payload = serde_json::json!({
            "file": file_path,
            "change_type": change_type,
        });
        self.record(
            memory_session_id,
            EventKind::FileChange,
            Some(file_path),
            Some(&payload.to_string()),
        )
    }

    /// Record a free-form note event.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_note(&self, memory_session_id: &Uuid, note: &str) -> Result<i64> {
        self.record(memory_session_id, EventKind::Note, Some(note), None)
    }

    fn record(
        &self,
        memory_session_id: &Uuid,
        kind: EventKind,
        title: Option<&str>,
        payload_json: Option<&str>,
    ) -> Result<i64> {
        let event = SessionEvent {
            event_id: None,
            memory_session_id: *memory_session_id,
            timestamp: Utc::now(),
            kind,
            title: title.map(String::from),
            payload_json: payload_json.map(String::from),
            redaction_level: self.default_redaction,
        };
        self.db.insert_event(&event)
    }

    fn maybe_redact(&self, text: &str) -> String {
        match self.default_redaction {
            RedactionLevel::None => text.to_owned(),
            RedactionLevel::Partial => {
                if text.len() > 200 {
                    format!("{}... [truncated]", &text[..200])
                } else {
                    text.to_owned()
                }
            }
            RedactionLevel::Full => "[REDACTED]".to_owned(),
        }
    }
}
