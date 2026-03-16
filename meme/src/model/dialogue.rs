//! Dialogue input type — raw conversation turns before compression.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single dialogue turn from a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dialogue {
    /// Sequential dialogue identifier.
    pub id: u64,
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
    pub fn new(id: u64, speaker: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id,
            speaker: speaker.into(),
            content: content.into(),
            timestamp: None,
        }
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }
}

impl std::fmt::Display for Dialogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ts) = self.timestamp {
            write!(
                f,
                "[{}] {}: {}",
                ts.format("%+"),
                self.speaker,
                self.content
            )
        } else {
            write!(f, "{}: {}", self.speaker, self.content)
        }
    }
}
