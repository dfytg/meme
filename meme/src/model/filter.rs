//! Metadata filter for structured (symbolic) search.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
    fn is_empty_checks_all_fields() {
        assert!(MetadataFilter::default().is_empty());

        let now = Utc::now();
        let cases: Vec<MetadataFilter> = vec![
            MetadataFilter {
                persons: Some(vec!["Alice".into()]),
                ..Default::default()
            },
            MetadataFilter {
                location: Some("Tokyo".into()),
                ..Default::default()
            },
            MetadataFilter {
                entities: Some(vec!["OpenAI".into()]),
                ..Default::default()
            },
            MetadataFilter {
                timestamp_range: Some((now, now)),
                ..Default::default()
            },
        ];
        for f in &cases {
            assert!(!f.is_empty(), "expected non-empty for {f:?}");
        }
    }
}
