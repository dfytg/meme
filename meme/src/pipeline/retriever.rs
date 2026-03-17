//! Stage 3: Hybrid Retriever — Intent-Aware Retrieval Planning.
//!
//! Implements multi-view retrieval across semantic, lexical, and symbolic layers
//! with optional reflection-based refinement.

use std::collections::HashSet;
use std::sync::Arc;

use crate::config::PipelineConfig;
use crate::embedding::Embedder;
use crate::error::Result;
use crate::llm::{ChatOptions, LlmClient, Message, extract_json_from_text, prompt};
use crate::model::{MemoryEntry, MetadataFilter};
use crate::store::{Scope, VectorStore};

/// Hybrid retriever that combines semantic, lexical, and symbolic search
/// with LLM-driven intent analysis and reflection.
pub struct HybridRetriever {
    llm: Arc<LlmClient>,
    store: Arc<VectorStore>,
    embedder: Arc<Embedder>,
    scope: Scope,
    semantic_top_k: usize,
    keyword_top_k: usize,
    structured_top_k: usize,
    enable_planning: bool,
    enable_reflection: bool,
    max_reflection_rounds: usize,
    max_retrieval_workers: usize,
}

impl std::fmt::Debug for HybridRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridRetriever")
            .field("semantic_top_k", &self.semantic_top_k)
            .field("enable_planning", &self.enable_planning)
            .field("enable_reflection", &self.enable_reflection)
            .finish()
    }
}

impl HybridRetriever {
    /// Create a new hybrid retriever.
    #[must_use]
    pub const fn new(
        llm: Arc<LlmClient>,
        store: Arc<VectorStore>,
        embedder: Arc<Embedder>,
        pipeline_cfg: &PipelineConfig,
        max_retrieval_workers: usize,
        scope: Scope,
    ) -> Self {
        Self {
            llm,
            store,
            embedder,
            scope,
            semantic_top_k: pipeline_cfg.semantic_top_k,
            keyword_top_k: pipeline_cfg.keyword_top_k,
            structured_top_k: pipeline_cfg.structured_top_k,
            enable_planning: pipeline_cfg.enable_planning,
            enable_reflection: pipeline_cfg.enable_reflection,
            max_reflection_rounds: pipeline_cfg.max_reflection_rounds,
            max_retrieval_workers,
        }
    }

    /// Execute retrieval with planning and optional reflection.
    ///
    /// # Errors
    ///
    /// Returns an error if any retrieval step fails.
    #[tracing::instrument(skip(self))]
    pub async fn retrieve(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        if self.enable_planning {
            self.retrieve_with_planning(query).await
        } else {
            self.semantic_search(query).await
        }
    }

    #[tracing::instrument(skip(self))]
    async fn retrieve_with_planning(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        // Single LLM call: unified query analysis + search planning.
        let plan = self.plan_query(query).await?;

        // Extract search queries from plan.
        let mut search_queries: Vec<String> = plan["search_queries"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if !search_queries.iter().any(|q| q == query) {
            search_queries.insert(0, query.to_owned());
        }
        search_queries.truncate(4);
        tracing::info!(count = search_queries.len(), "targeted queries");

        // Execute all three search views in parallel.
        let (semantic_results, keyword_results, structured_results) = tokio::join!(
            self.execute_semantic_searches(&search_queries),
            self.keyword_search(query, &plan),
            self.structured_search(&plan),
        );

        let mut all_results = semantic_results?;
        all_results.extend(keyword_results?);
        all_results.extend(structured_results?);

        let mut merged = deduplicate(all_results);
        tracing::info!(count = merged.len(), "unique results after merge");

        if self.enable_reflection {
            merged = self.reflect(query, merged, &plan).await?;
        }

        Ok(merged)
    }

    async fn semantic_search(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        let query_vec = self.embedder.encode_query(query).await?;
        self.store
            .semantic_search(&query_vec, self.semantic_top_k, &self.scope)
            .await
    }

    async fn keyword_search(
        &self,
        query: &str,
        analysis: &serde_json::Value,
    ) -> Result<Vec<MemoryEntry>> {
        let keywords: Vec<String> = analysis["keywords"].as_array().map_or_else(
            || vec![query.to_owned()],
            |a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            },
        );

        if keywords.is_empty() {
            return Ok(Vec::new());
        }
        self.store
            .keyword_search(&keywords, self.keyword_top_k, &self.scope)
            .await
    }

    async fn structured_search(&self, analysis: &serde_json::Value) -> Result<Vec<MemoryEntry>> {
        let persons: Option<Vec<String>> = analysis["persons"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .filter(|v: &Vec<String>| !v.is_empty());

        let location = analysis["location"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .map(String::from);

        let entities: Option<Vec<String>> = analysis["entities"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .filter(|v: &Vec<String>| !v.is_empty());

        let timestamp_range = analysis["time_expression"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .and_then(|expr| parse_time_range(expr, chrono::Utc::now()));

        let filter = MetadataFilter {
            persons,
            location,
            entities,
            timestamp_range,
        };

        if filter.is_empty() {
            return Ok(Vec::new());
        }

        self.store
            .structured_search(&filter, self.structured_top_k, &self.scope)
            .await
    }

    async fn execute_semantic_searches(&self, queries: &[String]) -> Result<Vec<MemoryEntry>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let mut handles = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_retrieval_workers));

        for query in queries {
            let embedder = Arc::clone(&self.embedder);
            let store = Arc::clone(&self.store);
            let top_k = self.semantic_top_k;
            let q = query.clone();
            let sem = Arc::clone(&semaphore);
            let scope = self.scope.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await;
                let query_vec = embedder.encode_query(&q).await?;
                store.semantic_search(&query_vec, top_k, &scope).await
            }));
        }

        let mut all_results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(results)) => all_results.extend(results),
                Ok(Err(e)) => tracing::warn!(error = %e, "parallel search failed"),
                Err(e) => tracing::warn!(error = %e, "search task panicked"),
            }
        }
        Ok(all_results)
    }

    async fn plan_query(&self, query: &str) -> Result<serde_json::Value> {
        let prompt = prompt::query_plan(query);
        let messages = vec![
            Message::system(
                "You are a query analysis and retrieval planning assistant. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.2,
            json_mode: true,
        };

        match self.llm.chat(&messages, &opts).await {
            Ok(response) => extract_json_from_text(&response),
            Err(e) => {
                tracing::warn!(error = %e, "query planning failed, using fallback");
                Ok(serde_json::json!({
                    "keywords": [query],
                    "persons": [],
                    "time_expression": null,
                    "location": null,
                    "entities": [],
                    "required_info": [],
                    "search_queries": [query]
                }))
            }
        }
    }

    async fn reflect(
        &self,
        query: &str,
        initial_results: Vec<MemoryEntry>,
        plan: &serde_json::Value,
    ) -> Result<Vec<MemoryEntry>> {
        let mut current = initial_results;
        let required_info = serde_json::to_string(&plan["required_info"]).unwrap_or_default();

        for round in 0..self.max_reflection_rounds {
            if current.is_empty() {
                tracing::info!(round = round + 1, "no results, stopping reflection");
                break;
            }

            let context_str = prompt::format_contexts_compact(&current);
            let assessment = self
                .check_completeness(query, &context_str, &required_info)
                .await?;

            let status = assessment["assessment"].as_str().unwrap_or("incomplete");

            if status == "complete" {
                tracing::info!(round = round + 1, "information complete");
                break;
            }

            tracing::info!(
                round = round + 1,
                "information incomplete, generating additional queries"
            );

            let additional_queries = self
                .generate_missing_queries(query, &context_str, &required_info)
                .await?;

            if additional_queries.is_empty() {
                break;
            }

            let additional_results = self.execute_semantic_searches(&additional_queries).await?;
            current.extend(additional_results);
            current = deduplicate(current);

            tracing::info!(
                round = round + 1,
                total = current.len(),
                "reflection round complete"
            );
        }

        Ok(current)
    }

    async fn check_completeness(
        &self,
        query: &str,
        context_str: &str,
        required_info: &str,
    ) -> Result<serde_json::Value> {
        let prompt = prompt::completeness_check(query, context_str, required_info);
        let messages = vec![
            Message::system(
                "You are an information completeness evaluator. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.1,
            json_mode: true,
        };
        let response = self.llm.chat(&messages, &opts).await?;
        extract_json_from_text(&response)
    }

    async fn generate_missing_queries(
        &self,
        query: &str,
        context_str: &str,
        required_info: &str,
    ) -> Result<Vec<String>> {
        let prompt = prompt::missing_info_queries(query, context_str, required_info);
        let messages = vec![
            Message::system(
                "You are a missing information query generator. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.3,
            json_mode: true,
        };
        let response = self.llm.chat(&messages, &opts).await?;
        let result = extract_json_from_text(&response)?;

        Ok(result["targeted_queries"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default())
    }
}

fn deduplicate(entries: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
    let mut seen = HashSet::new();
    entries.into_iter().filter(|e| seen.insert(e.id)).collect()
}

static RE_LAST_N_DAYS: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"last\s+(\d+)\s+days?").expect("valid regex"));

/// Parse a time expression into a `(start, end)` datetime range.
///
/// Supports:
/// - Relative: "last week", "yesterday", "last month", "last 3 days"
/// - ISO 8601: "2025-11-15T14:00:00Z"
/// - Date only: "2025-11-15", "November 15"
fn parse_time_range(
    expr: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)> {
    use chrono::{Duration, NaiveDate, TimeZone, Utc};

    let lower = expr.trim().to_lowercase();

    // Relative expressions.
    if lower.contains("yesterday") {
        let start = (now - Duration::days(1))
            .date_naive()
            .and_hms_opt(0, 0, 0)?;
        let end = (now - Duration::days(1))
            .date_naive()
            .and_hms_opt(23, 59, 59)?;
        return Some((Utc.from_utc_datetime(&start), Utc.from_utc_datetime(&end)));
    }
    if lower.contains("today") {
        let start = now.date_naive().and_hms_opt(0, 0, 0)?;
        let end = now.date_naive().and_hms_opt(23, 59, 59)?;
        return Some((Utc.from_utc_datetime(&start), Utc.from_utc_datetime(&end)));
    }
    if lower.contains("last week") || lower.contains("past week") {
        let start = now - Duration::days(7);
        return Some((start, now));
    }
    if lower.contains("last month") || lower.contains("past month") {
        let start = now - Duration::days(30);
        return Some((start, now));
    }

    if let Some(caps) = RE_LAST_N_DAYS.captures(&lower)
        && let Ok(n) = caps[1].parse::<i64>()
    {
        let start = now - Duration::days(n);
        return Some((start, now));
    }

    // Try ISO 8601 datetime parse.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(expr.trim()) {
        let dt = dt.with_timezone(&Utc);
        let start = dt.date_naive().and_hms_opt(0, 0, 0)?;
        let end = dt.date_naive().and_hms_opt(23, 59, 59)?;
        return Some((Utc.from_utc_datetime(&start), Utc.from_utc_datetime(&end)));
    }

    // Try "YYYY-MM-DD" date parse.
    if let Ok(date) = NaiveDate::parse_from_str(expr.trim(), "%Y-%m-%d") {
        let start = date.and_hms_opt(0, 0, 0)?;
        let end = date.and_hms_opt(23, 59, 59)?;
        return Some((Utc.from_utc_datetime(&start), Utc.from_utc_datetime(&end)));
    }

    // Try NaiveDateTime without timezone.
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(expr.trim(), "%Y-%m-%dT%H:%M:%S") {
        let start = ndt.date().and_hms_opt(0, 0, 0)?;
        let end = ndt.date().and_hms_opt(23, 59, 59)?;
        return Some((Utc.from_utc_datetime(&start), Utc.from_utc_datetime(&end)));
    }

    None
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;

    fn fixed_now() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 6, 15, 12, 0, 0).unwrap()
    }

    #[test]
    fn parse_time_range_yesterday() {
        let now = fixed_now();
        let (start, end) = parse_time_range("yesterday", now).unwrap();
        assert_eq!(
            start.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 6, 14).unwrap()
        );
        assert_eq!(
            end.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 6, 14).unwrap()
        );
    }

    #[test]
    fn parse_time_range_today() {
        let now = fixed_now();
        let (start, end) = parse_time_range("today", now).unwrap();
        assert_eq!(start.date_naive(), now.date_naive());
        assert_eq!(end.date_naive(), now.date_naive());
    }

    #[test]
    fn parse_time_range_last_week() {
        let now = fixed_now();
        let (start, end) = parse_time_range("last week", now).unwrap();
        assert_eq!((end - start).num_days(), 7);
    }

    #[test]
    fn parse_time_range_last_month() {
        let now = fixed_now();
        let (start, end) = parse_time_range("last month", now).unwrap();
        assert_eq!((end - start).num_days(), 30);
    }

    #[test]
    fn parse_time_range_last_n_days() {
        let now = fixed_now();
        let (start, end) = parse_time_range("last 5 days", now).unwrap();
        assert_eq!((end - start).num_days(), 5);
    }

    #[test]
    fn parse_time_range_iso_datetime() {
        let now = fixed_now();
        let (start, end) = parse_time_range("2025-11-15T14:00:00Z", now).unwrap();
        assert_eq!(
            start.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 11, 15).unwrap()
        );
        assert_eq!(
            end.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 11, 15).unwrap()
        );
    }

    #[test]
    fn parse_time_range_date_only() {
        let now = fixed_now();
        let (start, end) = parse_time_range("2025-11-15", now).unwrap();
        assert_eq!(
            start.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 11, 15).unwrap()
        );
        assert_eq!(
            end.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 11, 15).unwrap()
        );
    }

    #[test]
    fn parse_time_range_naive_datetime() {
        let now = fixed_now();
        let (start, end) = parse_time_range("2025-11-15T14:00:00", now).unwrap();
        assert_eq!(
            start.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 11, 15).unwrap()
        );
        assert_eq!(
            end.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2025, 11, 15).unwrap()
        );
    }

    #[test]
    fn parse_time_range_invalid() {
        let now = fixed_now();
        assert!(parse_time_range("", now).is_none());
        assert!(parse_time_range("some random text", now).is_none());
        assert!(parse_time_range("null", now).is_none());
    }

    #[test]
    fn parse_time_range_past_week_alias() {
        let now = fixed_now();
        let result = parse_time_range("past week", now);
        assert!(result.is_some());
    }

    #[test]
    fn deduplicate_removes_dups() {
        let e1 = MemoryEntry::new("fact one");
        let e2 = MemoryEntry::new("fact two");
        let e1_dup = e1.clone();
        let results = deduplicate(vec![e1.clone(), e2.clone(), e1_dup]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, e1.id);
        assert_eq!(results[1].id, e2.id);
    }

    #[test]
    fn deduplicate_empty() {
        let results = deduplicate(Vec::new());
        assert!(results.is_empty());
    }

    #[test]
    fn deduplicate_no_dups() {
        let entries = vec![
            MemoryEntry::new("a"),
            MemoryEntry::new("b"),
            MemoryEntry::new("c"),
        ];
        assert_eq!(deduplicate(entries).len(), 3);
    }
}
