//! Core memory — the fundamental unit of stored knowledge.

use std::fmt;
use std::hash::{Hash, Hasher};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A self-contained, unambiguous memory unit with multi-view indexing.
///
/// Each memory stores lossless content (no pronouns, absolute timestamps)
/// along with three indexing layers:
/// - **Semantic**: the `content` text is embedded as a dense vector
/// - **Lexical**: `keywords` enable BM25-style exact matching
/// - **Symbolic**: structured metadata (`timestamp`, `location`, `persons`, `entities`, `topic`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier.
    pub id: Uuid,
    /// Lossless content — complete, independent, no pronouns, absolute time.
    pub content: String,
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
    /// When this memory was first stored.
    pub created_at: DateTime<Utc>,
    /// When this memory was last modified (`None` if never updated).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    /// Namespace for memory isolation (opaque, caller-defined).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl PartialEq for Memory {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Memory {}

impl Hash for Memory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", &self.id.to_string()[..8], self.content)
    }
}

impl Memory {
    /// Create a new memory with an auto-generated UUID.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            keywords: Vec::new(),
            timestamp: None,
            location: None,
            persons: Vec::new(),
            entities: Vec::new(),
            topic: None,
            created_at: Utc::now(),
            updated_at: None,
            namespace: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_uuid() {
        let e = Memory::new("test fact");
        assert_eq!(e.content, "test fact");
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
        let e1 = Memory::new("a");
        let e2 = Memory::new("b");
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn serde_roundtrip() {
        let mut e = Memory::new("Alice met Bob at 2pm");
        e.keywords = vec!["meeting".into(), "Alice".into()];
        e.persons = vec!["Alice".into(), "Bob".into()];
        e.topic = Some("schedule".into());
        let json = serde_json::to_string(&e).unwrap();
        let e2: Memory = serde_json::from_str(&json).unwrap();
        assert_eq!(e.id, e2.id);
        assert_eq!(e.content, e2.content);
        assert_eq!(e.keywords, e2.keywords);
        assert_eq!(e.persons, e2.persons);
        assert_eq!(e.topic, e2.topic);
    }

    #[test]
    fn hash_by_id() {
        use std::collections::HashSet;
        let e1 = Memory::new("a");
        let e1_clone = e1.clone();
        let e2 = Memory::new("b");
        let mut set = HashSet::new();
        set.insert(e1);
        assert!(!set.insert(e1_clone));
        assert!(set.insert(e2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn display_format() {
        let e = Memory::new("test content");
        let display = format!("{e}");
        assert!(display.contains("test content"));
        assert!(display.starts_with('['));
    }
}
