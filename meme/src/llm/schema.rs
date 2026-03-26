//! Strongly-typed LLM response schemas.
//!
//! Each struct maps 1:1 to an `OpenAI` Structured Outputs JSON schema.
//! Used with [`LlmClient::chat_structured`] for type-safe deserialization.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use uuid::Uuid;

use crate::model::Memory;

/// Response from the extraction prompt (Stage 1).
#[derive(Debug, Deserialize)]
pub struct ExtractionResponse {
    /// Extracted memory entries.
    #[serde(default)]
    pub entries: Vec<ExtractedEntry>,
}

/// A single extracted entry from dialogue compression.
#[derive(Debug, Deserialize)]
pub struct ExtractedEntry {
    /// Complete, unambiguous restatement.
    pub lossless_restatement: String,
    /// Core keywords for lexical matching.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// ISO 8601 timestamp (if applicable).
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub timestamp: Option<String>,
    /// Location description.
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub location: Option<String>,
    /// People mentioned.
    #[serde(default)]
    pub persons: Vec<String>,
    /// Entities mentioned (companies, products, etc.).
    #[serde(default)]
    pub entities: Vec<String>,
    /// Topic phrase.
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub topic: Option<String>,
}

impl ExtractedEntry {
    /// Convert into a [`Memory`] with a fresh UUID.
    #[must_use]
    pub fn into_memory(self) -> Option<Memory> {
        if self.lossless_restatement.is_empty() {
            return None;
        }
        let timestamp = self.timestamp.as_deref().and_then(parse_timestamp);
        Some(Memory {
            id: Uuid::new_v4(),
            content: self.lossless_restatement,
            keywords: self.keywords,
            timestamp,
            location: self.location,
            persons: self.persons,
            entities: self.entities,
            topic: self.topic,
            user_id: None,
            session_id: None,
        })
    }
}

/// Response from the query plan prompt (Stage 3).
#[derive(Debug, Default, Deserialize)]
pub struct QueryPlan {
    /// Core keywords for lexical search.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Person names mentioned in the query.
    #[serde(default)]
    pub persons: Vec<String>,
    /// Time expression (e.g. "last 7 days", "2025-03-17").
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub time_expression: Option<String>,
    /// Location filter.
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub location: Option<String>,
    /// Entity names.
    #[serde(default)]
    pub entities: Vec<String>,
    /// What information is needed for a complete answer.
    #[serde(default)]
    pub required_info: Vec<String>,
    /// Targeted semantic search queries.
    #[serde(default)]
    pub search_queries: Vec<String>,
}

/// Response from the reconcile prompt.
#[derive(Debug, Deserialize)]
pub struct ReconcileResponse {
    /// Per-entry reconciliation decisions.
    #[serde(default)]
    pub actions: Vec<ReconcileAction>,
}

/// A single reconciliation decision.
#[derive(Debug, Deserialize)]
pub struct ReconcileAction {
    /// Index into the new entries array.
    #[serde(default, deserialize_with = "deserialize_index")]
    pub new_index: Option<usize>,
    /// One of: "add", "update", "delete", "noop".
    #[serde(default = "default_action")]
    pub action: String,
    /// Index into the existing entries array (for update/delete).
    #[serde(default, deserialize_with = "deserialize_index")]
    pub existing_index: Option<usize>,
}

/// Response from the answer generation prompt.
#[derive(Debug, Deserialize)]
pub struct AnswerResponse {
    /// The generated answer text.
    #[serde(default)]
    pub answer: String,
}

/// Response from the completeness check prompt.
#[derive(Debug, Deserialize)]
pub struct CompletenessResponse {
    /// "complete" or "incomplete".
    #[serde(default)]
    pub assessment: String,
}

/// Response from the missing-info queries prompt.
#[derive(Debug, Deserialize)]
pub struct MissingQueriesResponse {
    /// Additional targeted search queries.
    #[serde(default)]
    pub targeted_queries: Vec<String>,
}

/// Response from the metadata re-extraction prompt.
#[derive(Debug, Deserialize)]
pub struct ReExtractResponse {
    /// Core keywords.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// ISO 8601 timestamp.
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub timestamp: Option<String>,
    /// Location.
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub location: Option<String>,
    /// People mentioned.
    #[serde(default)]
    pub persons: Vec<String>,
    /// Entities mentioned.
    #[serde(default)]
    pub entities: Vec<String>,
    /// Topic phrase.
    #[serde(default, deserialize_with = "deserialize_nullable_str")]
    pub topic: Option<String>,
}

impl ReExtractResponse {
    /// Apply extracted metadata onto an existing [`Memory`].
    pub fn apply_to(self, entry: &mut Memory) {
        entry.keywords = self.keywords;
        entry.persons = self.persons;
        entry.entities = self.entities;
        entry.location = self.location;
        entry.topic = self.topic;
        if let Some(ts) = self.timestamp.as_deref().and_then(parse_timestamp) {
            entry.timestamp = Some(ts);
        }
    }
}

fn default_action() -> String {
    "add".to_owned()
}

/// Parse a timestamp string in RFC 3339 or `%Y-%m-%dT%H:%M:%S` format.
fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
                .ok()
                .map(|ndt| ndt.and_utc())
        })
}

/// Deserialize a JSON value that may be `null`, `"null"`, or `""` as `None`.
fn deserialize_nullable_str<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    Ok(opt.filter(|s| !s.is_empty() && s != "null"))
}

/// Deserialize an index that may be an integer or a string digit.
fn deserialize_index<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    Ok(val.and_then(|v| {
        v.as_u64()
            .map(|n| n as usize)
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    }))
}
