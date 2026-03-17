//! Memory lifecycle events — tracks add/update/delete history.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of memory lifecycle event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// A new memory was created.
    Add,
    /// An existing memory was updated.
    Update,
    /// An existing memory was deleted.
    Delete,
}

impl EventType {
    /// String representation for storage and display.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }

    /// Parse from a database string, defaulting to [`EventType::Add`] for unknown values.
    #[must_use]
    pub fn from_db_str(s: &str) -> Self {
        match s {
            "update" => Self::Update,
            "delete" => Self::Delete,
            _ => Self::Add,
        }
    }
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A recorded history event for a single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    /// Auto-generated event ID.
    pub id: Uuid,
    /// The memory entry this event relates to.
    pub memory_id: Uuid,
    /// The type of operation performed.
    pub event_type: EventType,
    /// Previous content (for updates and deletes).
    pub old_content: Option<String>,
    /// New content (for adds and updates).
    pub new_content: Option<String>,
    /// Timestamp of the event.
    pub timestamp: DateTime<Utc>,
}
