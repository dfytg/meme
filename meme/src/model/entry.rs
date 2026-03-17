//! Core memory entry — the fundamental unit of stored knowledge.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A self-contained, unambiguous memory unit with multi-view indexing.
///
/// Each entry stores a lossless restatement (no pronouns, absolute timestamps)
/// along with three indexing layers:
/// - **Semantic**: the `restatement` text is embedded as a dense vector
/// - **Lexical**: `keywords` enable BM25-style exact matching
/// - **Symbolic**: structured metadata (`timestamp`, `location`, `persons`, `entities`, `topic`)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: Uuid,
    /// Lossless restatement — complete, independent, no pronouns, absolute time.
    pub restatement: String,
    /// Core keywords for BM25 matching.
    pub keywords: Vec<String>,
    /// ISO 8601 timestamp (if applicable).
    pub timestamp: Option<DateTime<Utc>>,
    /// Location description.
    pub location: Option<String>,
    /// People mentioned.
    pub persons: Vec<String>,
    /// Entities mentioned (companies, products, etc.).
    pub entities: Vec<String>,
    /// Topic phrase.
    pub topic: Option<String>,
    /// Owner user identifier for multi-tenant isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Session identifier for multi-session isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl MemoryEntry {
    /// Create a new entry with an auto-generated UUID.
    #[must_use]
    pub fn new(restatement: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            restatement: restatement.into(),
            keywords: Vec::new(),
            timestamp: None,
            location: None,
            persons: Vec::new(),
            entities: Vec::new(),
            topic: None,
            user_id: None,
            session_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_uuid() {
        let e = MemoryEntry::new("test fact");
        assert_eq!(e.restatement, "test fact");
        assert!(!e.id.is_nil());
        assert!(e.keywords.is_empty());
        assert!(e.timestamp.is_none());
        assert!(e.location.is_none());
        assert!(e.persons.is_empty());
        assert!(e.entities.is_empty());
        assert!(e.topic.is_none());
    }

    #[test]
    fn unique_ids() {
        let e1 = MemoryEntry::new("a");
        let e2 = MemoryEntry::new("b");
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn serde_roundtrip() {
        let mut e = MemoryEntry::new("Alice met Bob at 2pm");
        e.keywords = vec!["meeting".into(), "Alice".into()];
        e.persons = vec!["Alice".into(), "Bob".into()];
        e.topic = Some("schedule".into());
        let json = serde_json::to_string(&e).unwrap();
        let e2: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }
}
