//! Event collector with sensitive-data redaction.
//!
//! Captures session events (messages, tool uses, file changes, notes) and applies
//! regex-driven redaction to strip API keys, JWT tokens, passwords, and other secrets.

use std::sync::LazyLock;

use chrono::Utc;
use uuid::Uuid;

use crate::error::Result;
use crate::model::{EventKind, RedactionLevel, SessionEvent};
use crate::store::SqliteStore;

static SENSITIVE_PATTERNS: LazyLock<Vec<(regex::Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (
            regex::Regex::new(r"(sk-[a-zA-Z0-9]{20,})").expect("valid"),
            "[REDACTED_API_KEY]",
        ),
        (
            regex::Regex::new(r"(key-[a-zA-Z0-9]{20,})").expect("valid"),
            "[REDACTED_KEY]",
        ),
        (
            regex::Regex::new(r#"(token["\s:=]+["']?[a-zA-Z0-9_\-\.]{20,}["']?)"#).expect("valid"),
            "[REDACTED_TOKEN]",
        ),
        (
            regex::Regex::new(r#"(password["\s:=]+["']?[^\s"']{4,}["']?)"#).expect("valid"),
            "[REDACTED_PASSWORD]",
        ),
        (
            regex::Regex::new(r"(Bearer\s+[a-zA-Z0-9_\-\.]+)").expect("valid"),
            "Bearer [REDACTED]",
        ),
        (
            regex::Regex::new(r#"(Authorization["\s:]+["']?[^\s"']+["']?)"#).expect("valid"),
            "Authorization: [REDACTED]",
        ),
        (
            regex::Regex::new(r"([A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")
                .expect("valid"),
            "[REDACTED_JWT]",
        ),
    ]
});

static SENSITIVE_FILE_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)(\.env|credentials|secret|secrets|password|token|\.pem|\.key|\.p12|\.pfx|id_rsa|id_dsa|\.npmrc|\.aws|\.gcp|\.azure)",
    )
    .expect("valid")
});

/// Detects and redacts sensitive content from text using regex patterns.
#[derive(Debug, Clone, Copy)]
pub struct RedactionFilter;

impl RedactionFilter {
    /// Apply all redaction patterns to the input text.
    ///
    /// Returns the redacted text and the level of redaction applied.
    #[must_use]
    pub fn redact(text: &str) -> (String, RedactionLevel) {
        if text.is_empty() {
            return (String::new(), RedactionLevel::None);
        }

        let mut redacted = text.to_owned();
        let mut level = RedactionLevel::None;

        for (pattern, replacement) in SENSITIVE_PATTERNS.iter() {
            if pattern.is_match(&redacted) {
                redacted = pattern.replace_all(&redacted, *replacement).into_owned();
                level = RedactionLevel::Partial;
            }
        }

        (redacted, level)
    }

    /// Check if a file path suggests sensitive content.
    #[must_use]
    pub fn should_redact_file(path: &str) -> bool {
        !path.is_empty() && SENSITIVE_FILE_PATTERN.is_match(path)
    }
}

/// Collects events during a session with automatic redaction.
pub struct EventCollector<'a> {
    db: &'a SqliteStore,
    redaction_enabled: bool,
    tool_output_max_len: usize,
}

impl std::fmt::Debug for EventCollector<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventCollector")
            .field("redaction_enabled", &self.redaction_enabled)
            .finish()
    }
}

impl<'a> EventCollector<'a> {
    /// Create a new event collector with redaction enabled.
    pub const fn new(db: &'a SqliteStore) -> Self {
        Self {
            db,
            redaction_enabled: true,
            tool_output_max_len: 2000,
        }
    }

    /// Create a collector with redaction disabled.
    pub const fn without_redaction(db: &'a SqliteStore) -> Self {
        Self {
            db,
            redaction_enabled: false,
            tool_output_max_len: 2000,
        }
    }

    /// Record a chat message event.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_message(
        &self,
        memory_session_id: &Uuid,
        role: &str,
        content: &str,
    ) -> Result<i64> {
        let (safe_content, redaction_level) = self.redact_text(content);
        let payload = serde_json::json!({
            "role": role,
            "content": safe_content,
        });
        self.record(
            memory_session_id,
            EventKind::Message,
            Some(role),
            Some(&payload.to_string()),
            redaction_level,
        )
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
        let (safe_input, input_level) = self.redact_text(tool_input);
        let (raw_output, output_level) = self.redact_text(tool_output);
        let safe_output = truncate(&raw_output, self.tool_output_max_len);

        let redaction_level = max_redaction_level(input_level, output_level);

        let payload = serde_json::json!({
            "tool_name": tool_name,
            "tool_input": safe_input,
            "tool_output": safe_output,
        });
        self.record(
            memory_session_id,
            EventKind::ToolUse,
            Some(tool_name),
            Some(&payload.to_string()),
            redaction_level,
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
        let (safe_path, redaction_level) = if RedactionFilter::should_redact_file(file_path) {
            ("[REDACTED_PATH]".to_owned(), RedactionLevel::Full)
        } else {
            (file_path.to_owned(), RedactionLevel::None)
        };

        let payload = serde_json::json!({
            "filepath": safe_path,
            "change_type": change_type,
        });
        self.record(
            memory_session_id,
            EventKind::FileChange,
            Some(change_type),
            Some(&payload.to_string()),
            redaction_level,
        )
    }

    /// Record a free-form note event.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_note(&self, memory_session_id: &Uuid, note: &str) -> Result<i64> {
        let (safe_note, redaction_level) = self.redact_text(note);
        let payload = serde_json::json!({ "note": safe_note });
        self.record(
            memory_session_id,
            EventKind::Note,
            Some("note"),
            Some(&payload.to_string()),
            redaction_level,
        )
    }

    fn record(
        &self,
        memory_session_id: &Uuid,
        kind: EventKind,
        title: Option<&str>,
        payload_json: Option<&str>,
        redaction_level: RedactionLevel,
    ) -> Result<i64> {
        let event = SessionEvent {
            event_id: None,
            memory_session_id: *memory_session_id,
            timestamp: Utc::now(),
            kind,
            title: title.map(String::from),
            payload_json: payload_json.map(String::from),
            redaction_level,
        };
        self.db.insert_event(&event)
    }

    fn redact_text(&self, text: &str) -> (String, RedactionLevel) {
        if self.redaction_enabled {
            RedactionFilter::redact(text)
        } else {
            (text.to_owned(), RedactionLevel::None)
        }
    }
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_owned()
    } else {
        let boundary = text.floor_char_boundary(max_len);
        format!("{}...", &text[..boundary])
    }
}

const fn max_redaction_level(a: RedactionLevel, b: RedactionLevel) -> RedactionLevel {
    match (a, b) {
        (RedactionLevel::Full, _) | (_, RedactionLevel::Full) => RedactionLevel::Full,
        (RedactionLevel::Partial, _) | (_, RedactionLevel::Partial) => RedactionLevel::Partial,
        _ => RedactionLevel::None,
    }
}
