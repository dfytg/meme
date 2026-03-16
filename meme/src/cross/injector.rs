//! Context injector — builds a token-budgeted context bundle at session start.
//!
//! Fills the bundle progressively in priority order:
//! 1. **Session summaries** (highest priority)
//! 2. **Observations** from recent sessions
//! 3. **Semantic search results** against the user's prompt (when provided)

use crate::error::Result;
use crate::model::{ContextBundle, CrossEntry, CrossObservation, SessionSummary};
use crate::store::SqliteStore;

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

impl<'a> ContextInjector<'a> {
    /// Create a new context injector.
    pub fn new(db: &'a SqliteStore, max_tokens: usize) -> Self {
        Self { db, max_tokens }
    }

    /// Build a context bundle for a session start.
    ///
    /// Fills three tiers in priority order within the token budget:
    /// 1. Recent session summaries
    /// 2. Recent observations (decisions, discoveries, changes)
    /// 3. Semantic search results (when `user_prompt` is provided)
    ///
    /// # Errors
    ///
    /// Returns an error if database queries fail.
    pub fn build_context(
        &self,
        project: &str,
        _user_prompt: Option<&str>,
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
        let _ = budget_remaining - tokens_used;
        total_tokens += tokens_used;
        tracing::debug!(
            packed = observations.len(),
            total = raw_observations.len(),
            tokens = tokens_used,
            "context injection: observations"
        );

        // Tier 3: Semantic search against user_prompt.
        // TODO: integrate VectorStore for semantic search when prompt is provided.
        let memory_entries: Vec<CrossEntry> = Vec::new();

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
