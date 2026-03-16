//! Context injector — builds a token-budgeted context bundle at session start.

use crate::error::Result;
use crate::model::ContextBundle;
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

    /// Build a context bundle for a session start, drawing from recent summaries
    /// and observations for the given project.
    ///
    /// # Errors
    ///
    /// Returns an error if database queries fail.
    pub fn build_context(&self, project: &str) -> Result<ContextBundle> {
        let summaries = self.db.get_recent_summaries(project, 10)?;

        let mut bundle = ContextBundle {
            session_summaries: summaries,
            timeline_observations: Vec::new(),
            memory_entries: Vec::new(),
            total_tokens_estimate: 0,
        };

        bundle.total_tokens_estimate = estimate_bundle_tokens(&bundle);

        // Trim if over budget.
        while bundle.total_tokens_estimate > self.max_tokens {
            if !bundle.memory_entries.is_empty() {
                bundle.memory_entries.pop();
            } else if !bundle.timeline_observations.is_empty() {
                bundle.timeline_observations.pop();
            } else if !bundle.session_summaries.is_empty() {
                bundle.session_summaries.pop();
            } else {
                break;
            }
            bundle.total_tokens_estimate = estimate_bundle_tokens(&bundle);
        }

        tracing::info!(
            summaries = bundle.session_summaries.len(),
            observations = bundle.timeline_observations.len(),
            entries = bundle.memory_entries.len(),
            tokens = bundle.total_tokens_estimate,
            "built context bundle"
        );

        Ok(bundle)
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

fn estimate_bundle_tokens(bundle: &ContextBundle) -> usize {
    let mut total = 0;

    for s in &bundle.session_summaries {
        if let Some(t) = &s.completed {
            total += estimate_tokens(t);
        }
        if let Some(t) = &s.learned {
            total += estimate_tokens(t);
        }
        if let Some(t) = &s.investigated {
            total += estimate_tokens(t);
        }
        if let Some(t) = &s.request {
            total += estimate_tokens(t);
        }
        if let Some(t) = &s.next_steps {
            total += estimate_tokens(t);
        }
    }

    for obs in &bundle.timeline_observations {
        total += estimate_tokens(&obs.title);
        if let Some(t) = &obs.subtitle {
            total += estimate_tokens(t);
        }
        if let Some(t) = &obs.narrative {
            total += estimate_tokens(t);
        }
    }

    for e in &bundle.memory_entries {
        total += estimate_tokens(&e.entry.restatement);
    }

    total
}
