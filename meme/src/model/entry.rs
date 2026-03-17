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
        }
    }
}

/// Filter criteria for symbolic (metadata) search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Filter by person names (any match).
    pub persons: Option<Vec<String>>,
    /// Filter by location (substring match).
    pub location: Option<String>,
    /// Filter by entity names (any match).
    pub entities: Option<Vec<String>>,
    /// Filter by timestamp range (inclusive).
    pub timestamp_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

impl MetadataFilter {
    /// Returns `true` if no filter criteria are set.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.persons.is_none()
            && self.location.is_none()
            && self.entities.is_none()
            && self.timestamp_range.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_entry_new_has_uuid() {
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
    fn memory_entry_unique_ids() {
        let e1 = MemoryEntry::new("a");
        let e2 = MemoryEntry::new("b");
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn metadata_filter_empty_default() {
        let f = MetadataFilter::default();
        assert!(f.is_empty());
    }

    #[test]
    fn metadata_filter_not_empty_with_persons() {
        let f = MetadataFilter {
            persons: Some(vec!["Alice".into()]),
            ..Default::default()
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn metadata_filter_not_empty_with_location() {
        let f = MetadataFilter {
            location: Some("Tokyo".into()),
            ..Default::default()
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn metadata_filter_not_empty_with_entities() {
        let f = MetadataFilter {
            entities: Some(vec!["OpenAI".into()]),
            ..Default::default()
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn metadata_filter_not_empty_with_timestamp() {
        let now = chrono::Utc::now();
        let f = MetadataFilter {
            timestamp_range: Some((now, now)),
            ..Default::default()
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn memory_entry_serde_roundtrip() {
        let mut e = MemoryEntry::new("Alice met Bob at 2pm");
        e.keywords = vec!["meeting".into(), "Alice".into()];
        e.persons = vec!["Alice".into(), "Bob".into()];
        e.topic = Some("schedule".into());
        let json = serde_json::to_string(&e).unwrap();
        let e2: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }
}
