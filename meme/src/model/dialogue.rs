//! Dialogue input type — raw conversation turns before compression.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single dialogue turn from a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dialogue {
    /// Speaker name.
    pub speaker: String,
    /// Dialogue content.
    pub content: String,
    /// Timestamp of the dialogue (ISO 8601).
    pub timestamp: Option<DateTime<Utc>>,
}

impl Dialogue {
    /// Create a new dialogue turn.
    #[must_use]
    pub fn new(speaker: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            content: content.into(),
            timestamp: None,
        }
    }

    /// Set the timestamp.
    #[must_use]
    pub const fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }
}

impl Dialogue {
    /// Format this dialogue for inclusion in an LLM extraction prompt.
    ///
    /// Includes timestamp when available for temporal anchoring.
    #[must_use]
    pub fn format_for_prompt(&self) -> String {
        self.timestamp.map_or_else(
            || format!("{}: {}", self.speaker, self.content),
            |ts| {
                format!(
                    "[{} {}] {}: {}",
                    ts.format("%d %B %Y"),
                    ts.format("%H:%M"),
                    self.speaker,
                    self.content,
                )
            },
        )
    }
}

impl std::fmt::Display for Dialogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.speaker, self.content)
    }
}
