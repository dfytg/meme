//! Context injector — builds a token-budgeted context bundle at session start.
//!
//! Fills the bundle progressively in priority order:
//! 1. **Session summaries** (highest priority)
//! 2. **Observations** from recent sessions
//! 3. **Semantic search results** against the user's prompt (when provided)

use uuid::Uuid;

use crate::embedding::Embedder;
use crate::error::Result;
use crate::model::{ContextBundle, CrossEntry, CrossObservation, SessionSummary};
use crate::store::{SqliteStore, VectorStore};

/// Builds a [`ContextBundle`] from past session data, constrained by a token budget.
pub struct ContextInjector<'a> {
    db: &'a SqliteStore,
    max_tokens: usize,
}

impl std::fmt::Debug for ContextInjector<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextInjector")
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[allow(clippy::future_not_send)]
impl<'a> ContextInjector<'a> {
    /// Create a new context injector.
    pub const fn new(db: &'a SqliteStore, max_tokens: usize) -> Self {
        Self { db, max_tokens }
    }

    /// Build a context bundle for a session start.
    ///
    /// Fills three tiers in priority order within the token budget:
    /// 1. Recent session summaries
    /// 2. Recent observations (decisions, discoveries, changes)
    /// 3. Semantic search results (when `user_prompt` is provided and stores available)
    ///
    /// # Errors
    ///
    /// Returns an error if database queries or vector search fails.
    pub async fn build_context(
        &self,
        project: &str,
        user_prompt: Option<&str>,
        vector_store: Option<&VectorStore>,
        embedder: Option<&Embedder>,
    ) -> Result<ContextBundle> {
        let mut budget_remaining = self.max_tokens;
        let mut total_tokens = 0usize;

        // Tier 1: Session summaries (highest priority).
        let raw_summaries = self.db.get_recent_summaries(project, 5)?;
        let (summaries, tokens_used) =
            budget_items(&raw_summaries, text_for_summary, budget_remaining);
        budget_remaining -= tokens_used;
        total_tokens += tokens_used;
        tracing::debug!(
            packed = summaries.len(),
            total = raw_summaries.len(),
            tokens = tokens_used,
            "context injection: summaries"
        );

        // Tier 2: Observations.
        let raw_observations = self.db.get_recent_observations(project, 20)?;
        let (observations, tokens_used) =
            budget_items(&raw_observations, text_for_observation, budget_remaining);
        budget_remaining -= tokens_used;
        total_tokens += tokens_used;
        tracing::debug!(
            packed = observations.len(),
            total = raw_observations.len(),
            tokens = tokens_used,
            "context injection: observations"
        );

        // Tier 3: Semantic search against user_prompt.
        let memory_entries =
            if let (Some(prompt), Some(store), Some(emb)) = (user_prompt, vector_store, embedder) {
                if budget_remaining > 0 && !prompt.is_empty() {
                    self.semantic_search_entries(prompt, store, emb, budget_remaining)
                        .await?
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

        if !memory_entries.is_empty() {
            let entry_tokens: usize = memory_entries
                .iter()
                .map(|e| estimate_tokens(&e.entry.restatement))
                .sum();
            total_tokens += entry_tokens;
            tracing::debug!(
                packed = memory_entries.len(),
                tokens = entry_tokens,
                "context injection: semantic entries"
            );
        }

        let bundle = ContextBundle {
            session_summaries: summaries,
            timeline_observations: observations,
            memory_entries,
            total_tokens_estimate: total_tokens,
        };

        tracing::info!(
            summaries = bundle.session_summaries.len(),
            observations = bundle.timeline_observations.len(),
            entries = bundle.memory_entries.len(),
            tokens = total_tokens,
            "context bundle built"
        );

        Ok(bundle)
    }

    async fn semantic_search_entries(
        &self,
        prompt: &str,
        store: &VectorStore,
        embedder: &Embedder,
        budget_remaining: usize,
    ) -> Result<Vec<CrossEntry>> {
        let query_vec = embedder.encode_query(prompt).await?;
        let top_k = 10;
        let entries = store.semantic_search(&query_vec, top_k).await?;

        let mut results = Vec::new();
        let mut tokens_used = 0usize;
        for entry in entries {
            let cost = estimate_tokens(&entry.restatement);
            if tokens_used + cost > budget_remaining {
                break;
            }
            tokens_used += cost;
            results.push(CrossEntry {
                entry,
                tenant_id: "default".to_owned(),
                memory_session_id: Uuid::nil(),
                source_kind: "semantic_search".to_owned(),
                source_id: None,
                importance: 1.0,
                valid_from: None,
                valid_to: None,
                superseded_by: None,
            });
        }
        Ok(results)
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

fn text_for_summary(s: &SessionSummary) -> String {
    let mut parts = Vec::new();
    if let Some(t) = &s.request {
        parts.push(format!("Request: {t}"));
    }
    if let Some(t) = &s.investigated {
        parts.push(format!("Investigated: {t}"));
    }
    if let Some(t) = &s.learned {
        parts.push(format!("Learned: {t}"));
    }
    if let Some(t) = &s.completed {
        parts.push(format!("Completed: {t}"));
    }
    if let Some(t) = &s.next_steps {
        parts.push(format!("Next steps: {t}"));
    }
    if parts.is_empty() {
        "Session summary available.".to_owned()
    } else {
        parts.join(" | ")
    }
}

fn text_for_observation(obs: &CrossObservation) -> String {
    let detail = obs
        .subtitle
        .as_deref()
        .or(obs.narrative.as_deref())
        .unwrap_or("");
    if detail.is_empty() {
        obs.title.clone()
    } else {
        format!("{}: {detail}", obs.title)
    }
}

/// Greedily pack items into a token budget.
///
/// Returns `(accepted_items, tokens_consumed)`.
fn budget_items<T: Clone>(
    items: &[T],
    text_fn: fn(&T) -> String,
    remaining_tokens: usize,
) -> (Vec<T>, usize) {
    let mut accepted = Vec::new();
    let mut consumed = 0usize;

    for item in items {
        let cost = estimate_tokens(&text_fn(item));
        if cost == 0 {
            accepted.push(item.clone());
            continue;
        }
        if consumed + cost > remaining_tokens {
            break;
        }
        accepted.push(item.clone());
        consumed += cost;
    }

    (accepted, consumed)
}

/// Render a context bundle as system-prompt text wrapped in XML tags.
#[must_use]
pub fn render_for_system_prompt(bundle: &ContextBundle, max_tokens: usize) -> String {
    let rendered = bundle.render(max_tokens);
    if rendered.is_empty() {
        return String::new();
    }
    format!(
        "<cross_session_memory>\n\
         The following is relevant context from previous sessions.\n\
         Use it to inform your responses but do not repeat it verbatim.\n\n\
         {rendered}\n\
         </cross_session_memory>"
    )
}
