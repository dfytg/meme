//! Stage 3: Hybrid Retriever — Intent-Aware Retrieval Planning.
//!
//! Implements multi-view retrieval across semantic, lexical, and symbolic layers
//! with optional reflection-based refinement.

use std::collections::HashSet;
use std::sync::Arc;

use crate::config::PipelineConfig;
use crate::embedding::EmbeddingProvider;
use crate::error::Result;
use crate::llm::client::{ChatOptions, LlmClient, Message, extract_json_from_text};
use crate::llm::prompt::Prompts;
use crate::model::{MemoryEntry, MetadataFilter};
use crate::store::VectorStore;

/// Hybrid retriever that combines semantic, lexical, and symbolic search
/// with LLM-driven intent analysis and reflection.
pub struct HybridRetriever {
    llm: Arc<dyn LlmClient>,
    store: Arc<dyn VectorStore>,
    embedding: Arc<dyn EmbeddingProvider>,
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
    pub fn new(
        llm: Arc<dyn LlmClient>,
        store: Arc<dyn VectorStore>,
        embedding: Arc<dyn EmbeddingProvider>,
        pipeline_cfg: &PipelineConfig,
        max_retrieval_workers: usize,
    ) -> Self {
        Self {
            llm,
            store,
            embedding,
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
    pub async fn retrieve(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        if self.enable_planning {
            self.retrieve_with_planning(query).await
        } else {
            self.semantic_search(query).await
        }
    }

    async fn retrieve_with_planning(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        tracing::info!(query, "analyzing information requirements");

        // Step 1: Analyze what information is needed.
        let plan = self.analyze_requirements(query).await?;

        // Step 2: Generate targeted queries.
        let search_queries = self.generate_targeted_queries(query, &plan).await?;
        tracing::info!(count = search_queries.len(), "generated targeted queries");

        // Step 3: Execute semantic searches (parallel).
        let mut all_results = self.execute_semantic_searches(&search_queries).await?;

        // Step 4: Keyword + structured searches.
        let analysis = self.analyze_query(query).await?;

        let keyword_results = self.keyword_search(query, &analysis).await?;
        tracing::info!(count = keyword_results.len(), "keyword search results");
        all_results.extend(keyword_results);

        let structured_results = self.structured_search(&analysis).await?;
        tracing::info!(
            count = structured_results.len(),
            "structured search results"
        );
        all_results.extend(structured_results);

        // Step 5: Merge and deduplicate.
        let mut merged = deduplicate(all_results);
        tracing::info!(count = merged.len(), "unique results after merge");

        // Step 6: Optional reflection.
        if self.enable_reflection {
            merged = self.reflect(query, merged, &plan).await?;
        }

        Ok(merged)
    }

    async fn semantic_search(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        let query_vec = self.embedding.encode_query(query).await?;
        self.store
            .semantic_search(&query_vec, self.semantic_top_k)
            .await
    }

    async fn keyword_search(
        &self,
        query: &str,
        analysis: &serde_json::Value,
    ) -> Result<Vec<MemoryEntry>> {
        let keywords: Vec<String> = analysis["keywords"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec![query.to_owned()]);

        if keywords.is_empty() {
            return Ok(Vec::new());
        }
        self.store
            .keyword_search(&keywords, self.keyword_top_k)
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
            .structured_search(&filter, self.structured_top_k)
            .await
    }

    async fn execute_semantic_searches(&self, queries: &[String]) -> Result<Vec<MemoryEntry>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let mut handles = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_retrieval_workers));

        for query in queries {
            let embedding = Arc::clone(&self.embedding);
            let store = Arc::clone(&self.store);
            let top_k = self.semantic_top_k;
            let q = query.clone();
            let sem = Arc::clone(&semaphore);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await;
                let query_vec = embedding.encode_query(&q).await?;
                store.semantic_search(&query_vec, top_k).await
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

    async fn analyze_requirements(&self, query: &str) -> Result<serde_json::Value> {
        let prompt = Prompts::information_requirements(query);
        let messages = vec![
            Message::system(
                "You are an intelligent information requirement analyst. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.2,
            json_mode: false,
        };
        let response = self.llm.chat(&messages, &opts).await?;
        extract_json_from_text(&response)
    }

    async fn generate_targeted_queries(
        &self,
        query: &str,
        plan: &serde_json::Value,
    ) -> Result<Vec<String>> {
        let plan_json = serde_json::to_string_pretty(plan).unwrap_or_default();
        let prompt = Prompts::targeted_queries(query, &plan_json);
        let messages = vec![
            Message::system(
                "You are a query generation specialist. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.3,
            json_mode: false,
        };

        let response = self.llm.chat(&messages, &opts).await?;
        let result = extract_json_from_text(&response)?;

        let mut queries: Vec<String> = result["queries"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec![query.to_owned()]);

        if !queries.iter().any(|q| q == query) {
            queries.insert(0, query.to_owned());
        }
        queries.truncate(4);
        Ok(queries)
    }

    async fn analyze_query(&self, query: &str) -> Result<serde_json::Value> {
        let prompt = Prompts::query_analysis(query);
        let messages = vec![
            Message::system(
                "You are a query analysis assistant. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.1,
            json_mode: false,
        };

        match self.llm.chat(&messages, &opts).await {
            Ok(response) => extract_json_from_text(&response),
            Err(e) => {
                tracing::warn!(error = %e, "query analysis failed, using fallback");
                Ok(serde_json::json!({
                    "keywords": [query],
                    "persons": [],
                    "time_expression": null,
                    "location": null,
                    "entities": []
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

            let context_str = Prompts::format_contexts_compact(&current);
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
        let prompt = Prompts::completeness_check(query, context_str, required_info);
        let messages = vec![
            Message::system(
                "You are an information completeness evaluator. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.1,
            json_mode: false,
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
        let prompt = Prompts::missing_info_queries(query, context_str, required_info);
        let messages = vec![
            Message::system(
                "You are a missing information query generator. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.3,
            json_mode: false,
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

    // "last N days" pattern.
    static RE_LAST_N_DAYS: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"last\s+(\d+)\s+days?").expect("valid"));
    if let Some(caps) = RE_LAST_N_DAYS.captures(&lower) {
        if let Ok(n) = caps[1].parse::<i64>() {
            let start = now - Duration::days(n);
            return Some((start, now));
        }
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
