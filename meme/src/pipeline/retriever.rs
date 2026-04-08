//! Stage 3: Hybrid Retriever — Intent-Aware Retrieval Planning.
//!
//! Implements multi-view retrieval across semantic, lexical, and symbolic layers
//! with optional reflection-based refinement.

use std::collections::HashSet;
use std::sync::Arc;

use crate::config::PipelineConfig;
use crate::embedding::Embedder;
use crate::error::Result;
use crate::llm::{
    ChatOptions, CompletenessResponse, LlmClient, Message, MissingQueriesResponse, QueryPlan,
    prompt,
};
use crate::model::{Memory, MetadataFilter};
use crate::store::VectorStore;

/// Hybrid retriever that combines semantic, lexical, and symbolic search
/// with LLM-driven intent analysis and reflection.
pub(crate) struct HybridRetriever {
    /// LLM client for query planning and reflection.
    llm: Arc<LlmClient>,
    /// Vector store backend.
    store: Arc<VectorStore>,
    /// Embedding model.
    embedder: Arc<Embedder>,
    /// Pipeline tuning parameters.
    config: PipelineConfig,
    /// Optional namespace filter.
    namespace: Option<String>,
    /// Optional ONNX cross-encoder reranker.
    #[cfg(feature = "onnx")]
    reranker: Option<Arc<crate::reranking::OnnxReranker>>,
}

impl std::fmt::Debug for HybridRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridRetriever")
            .field("semantic_top_k", &self.config.semantic_top_k)
            .field("enable_planning", &self.config.enable_planning)
            .finish_non_exhaustive()
    }
}

impl HybridRetriever {
    /// Create a new hybrid retriever.
    #[must_use]
    pub(crate) const fn new(
        llm: Arc<LlmClient>,
        store: Arc<VectorStore>,
        embedder: Arc<Embedder>,
        config: PipelineConfig,
        namespace: Option<String>,
        #[cfg(feature = "onnx")] reranker: Option<Arc<crate::reranking::OnnxReranker>>,
    ) -> Self {
        Self {
            llm,
            store,
            embedder,
            config,
            namespace,
            #[cfg(feature = "onnx")]
            reranker,
        }
    }

    /// Execute retrieval with planning and optional reflection.
    ///
    /// # Errors
    ///
    /// Returns an error if any retrieval step fails.
    #[tracing::instrument(skip(self))]
    pub(crate) async fn retrieve(&self, query: &str) -> Result<Vec<Memory>> {
        if self.config.enable_planning {
            self.retrieve_with_planning(query).await
        } else {
            self.semantic_search(query).await
        }
    }

    /// Full retrieval pipeline with query planning and optional reflection.
    #[tracing::instrument(skip(self))]
    async fn retrieve_with_planning(&self, query: &str) -> Result<Vec<Memory>> {
        let plan = self.plan_query(query).await?;

        let mut search_queries = plan.search_queries.clone();
        if !search_queries.iter().any(|q| q == query) {
            search_queries.insert(0, query.to_owned());
        }
        search_queries.truncate(5);
        tracing::info!(count = search_queries.len(), "targeted queries");

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

        if self.config.enable_reflection {
            merged = self.reflect(query, merged, &plan).await?;
        }

        #[cfg(feature = "onnx")]
        {
            merged = self.maybe_rerank(query, merged).await?;
        }

        Ok(merged)
    }

    /// Run a single semantic (embedding) search.
    async fn semantic_search(&self, query: &str) -> Result<Vec<Memory>> {
        let query_vec = self.embedder.encode_query(query).await?;
        self.store
            .semantic_search(&query_vec, self.config.semantic_top_k, self.ns())
            .await
    }

    /// Run keyword (LIKE) search against the store.
    async fn keyword_search(&self, query: &str, plan: &QueryPlan) -> Result<Vec<Memory>> {
        let keywords = if plan.keywords.is_empty() {
            vec![query.to_owned()]
        } else {
            plan.keywords.clone()
        };
        self.store
            .keyword_search(&keywords, self.config.keyword_top_k, self.ns())
            .await
    }

    /// Run structured metadata search (persons, location, time range).
    async fn structured_search(&self, plan: &QueryPlan) -> Result<Vec<Memory>> {
        let persons = Some(&plan.persons).filter(|v| !v.is_empty()).cloned();
        let entities = Some(&plan.entities).filter(|v| !v.is_empty()).cloned();
        let timestamp_range = plan
            .time_expression
            .as_deref()
            .and_then(|expr| parse_time_range(expr, chrono::Utc::now()));

        let filter = MetadataFilter {
            persons,
            location: plan.location.clone(),
            entities,
            timestamp_range,
        };
        if filter.is_empty() {
            return Ok(Vec::new());
        }
        self.store
            .structured_search(&filter, self.config.structured_top_k, self.ns())
            .await
    }

    /// Return the namespace filter as a `&str` slice.
    fn ns(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// Execute multiple semantic searches in parallel.
    async fn execute_semantic_searches(&self, queries: &[String]) -> Result<Vec<Memory>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let mut handles = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            self.config.max_retrieval_workers,
        ));

        for query in queries {
            let embedder = Arc::clone(&self.embedder);
            let store = Arc::clone(&self.store);
            let top_k = self.config.semantic_top_k;
            let q = query.clone();
            let sem = Arc::clone(&semaphore);
            let ns = self.namespace.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await;
                let query_vec = embedder.encode_query(&q).await?;
                store
                    .semantic_search(&query_vec, top_k, ns.as_deref())
                    .await
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

    /// Ask the LLM to decompose a query into a structured plan.
    async fn plan_query(&self, query: &str) -> Result<QueryPlan> {
        let prompt = prompt::query_plan(query);
        let messages = vec![
            Message::system(
                "You are a query analysis and retrieval planning assistant. Output valid JSON.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.2,
            json_mode: true,
        };

        match self
            .llm
            .chat_structured::<QueryPlan>(&messages, &opts)
            .await
        {
            Ok(plan) => Ok(plan),
            Err(e) => {
                tracing::warn!(error = %e, "query planning failed, using fallback");
                Ok(QueryPlan {
                    keywords: vec![query.to_owned()],
                    search_queries: vec![query.to_owned()],
                    ..QueryPlan::default()
                })
            }
        }
    }

    /// Iterative reflection: check completeness and fetch missing info.
    async fn reflect(
        &self,
        query: &str,
        initial_results: Vec<Memory>,
        plan: &QueryPlan,
    ) -> Result<Vec<Memory>> {
        let mut current = initial_results;
        let required_info = plan.required_info.join(", ");

        for round in 0..self.config.max_reflection_rounds {
            if current.is_empty() {
                tracing::info!(round = round + 1, "no results, stopping reflection");
                break;
            }

            let context_str = prompt::format_contexts_compact(&current);
            let assessment: CompletenessResponse = self
                .check_completeness(query, &context_str, &required_info)
                .await?;

            if assessment.assessment == "complete" {
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

    /// Ask the LLM whether the retrieved context covers the query.
    async fn check_completeness(
        &self,
        query: &str,
        context_str: &str,
        required_info: &str,
    ) -> Result<CompletenessResponse> {
        let prompt = prompt::completeness_check(query, context_str, required_info);
        let messages = vec![
            Message::system("You are an information completeness evaluator. Output valid JSON."),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.1,
            json_mode: true,
        };
        self.llm.chat_structured(&messages, &opts).await
    }

    /// Ask the LLM to produce additional search queries for missing info.
    async fn generate_missing_queries(
        &self,
        query: &str,
        context_str: &str,
        required_info: &str,
    ) -> Result<Vec<String>> {
        let prompt = prompt::missing_info_queries(query, context_str, required_info);
        let messages = vec![
            Message::system("You are a missing information query generator. Output valid JSON."),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.3,
            json_mode: true,
        };
        let resp: MissingQueriesResponse = self.llm.chat_structured(&messages, &opts).await?;
        Ok(resp.targeted_queries)
    }

    /// Apply reranker if configured, otherwise pass through unchanged.
    #[cfg(feature = "onnx")]
    async fn maybe_rerank(&self, query: &str, entries: Vec<Memory>) -> Result<Vec<Memory>> {
        if let Some(reranker) = &self.reranker {
            if entries.is_empty() {
                return Ok(entries);
            }
            let docs: Vec<&str> = entries.iter().map(|e| e.content.as_str()).collect();
            let top_n = self.config.rerank_top_n;
            let indices = reranker.rerank(query, &docs, top_n).await?;
            tracing::info!(
                before = entries.len(),
                after = indices.len(),
                "reranked results"
            );
            return Ok(indices
                .into_iter()
                .filter_map(|i| entries.get(i).cloned())
                .collect());
        }
        Ok(entries)
    }
}

/// Remove duplicate entries by `id`.
fn deduplicate(entries: Vec<Memory>) -> Vec<Memory> {
    let mut seen = HashSet::new();
    entries.into_iter().filter(|e| seen.insert(e.id)).collect()
}

/// Regex for "last N days" time expressions.
static RE_LAST_N_DAYS: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"last\s+(\d+)\s+days?").unwrap_or_else(|_| unreachable!())
});

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
        let e1 = Memory::new("fact one");
        let e2 = Memory::new("fact two");
        let e1_dup = e1.clone();
        let results = deduplicate(vec![e1.clone(), e2.clone(), e1_dup]);
        assert_eq!(results.len(), 2);
        assert_eq!(results.first().expect("2 results").id, e1.id);
        assert_eq!(results.get(1).expect("2 results").id, e2.id);
    }

    #[test]
    fn deduplicate_empty() {
        let results = deduplicate(Vec::new());
        assert!(results.is_empty());
    }

    #[test]
    fn deduplicate_no_dups() {
        let entries = vec![Memory::new("a"), Memory::new("b"), Memory::new("c")];
        assert_eq!(deduplicate(entries).len(), 3);
    }
}
