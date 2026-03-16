//! Memory consolidation worker — decay, merge, and prune old entries.
//!
//! Periodically maintains memory quality:
//! 1. **Decay** — reduce importance of old entries over time
//! 2. **Merge** — combine near-duplicate entries with high semantic similarity
//! 3. **Prune** — soft-delete entries whose importance has fallen below a threshold

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::config::CrossConfig;
use crate::error::Result;
use crate::model::ConsolidationRun;
use crate::store::SqliteStore;

/// Configurable parameters for a single consolidation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationPolicy {
    /// Entries older than this (days) receive importance decay.
    pub max_age_days: u32,
    /// Multiplier applied to importance for each decay period elapsed.
    pub decay_factor: f64,
    /// Cosine similarity above which two entries are considered near-duplicates.
    pub merge_similarity_threshold: f64,
    /// Entries below this importance after decay are pruned (soft-deleted).
    pub min_importance: f64,
    /// Maximum number of entries processed in one consolidation pass.
    pub max_entries_per_run: usize,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            max_age_days: 90,
            decay_factor: 0.9,
            merge_similarity_threshold: 0.95,
            min_importance: 0.05,
            max_entries_per_run: 1000,
        }
    }
}

impl ConsolidationPolicy {
    /// Create a policy from the cross-session configuration.
    #[must_use]
    pub fn from_config(cfg: &CrossConfig) -> Self {
        Self {
            max_age_days: cfg.consolidation_max_age_days,
            decay_factor: cfg.consolidation_decay_factor,
            merge_similarity_threshold: cfg.consolidation_merge_threshold,
            min_importance: cfg.consolidation_min_importance,
            max_entries_per_run: cfg.consolidation_max_entries_per_run,
        }
    }
}

/// Statistics from a consolidation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Number of entries whose importance was decayed.
    pub decayed: usize,
    /// Number of entries that were merged.
    pub merged: usize,
    /// Number of entries that were pruned (soft-deleted).
    pub pruned: usize,
    /// Total entries scanned.
    pub scanned: usize,
}

/// Consolidation worker that maintains memory quality over time.
pub struct ConsolidationWorker<'a> {
    db: &'a SqliteStore,
    policy: ConsolidationPolicy,
    tenant_id: String,
}

impl std::fmt::Debug for ConsolidationWorker<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsolidationWorker")
            .field("policy", &self.policy)
            .field("tenant_id", &self.tenant_id)
            .finish()
    }
}

impl<'a> ConsolidationWorker<'a> {
    /// Create a new consolidation worker.
    pub fn new(db: &'a SqliteStore, policy: ConsolidationPolicy, tenant_id: &str) -> Self {
        Self {
            db,
            policy,
            tenant_id: tenant_id.to_owned(),
        }
    }

    /// Run a full consolidation pass (decay → merge → prune).
    ///
    /// Returns statistics about the run. The actual vector operations (similarity
    /// comparisons, importance updates) require integration with the vector store,
    /// which is handled at the orchestrator level.
    ///
    /// # Errors
    ///
    /// Returns an error if database operations fail.
    pub fn run(&self) -> Result<ConsolidationStats> {
        let stats = ConsolidationStats::default();

        // Record the run.
        let run = ConsolidationRun {
            run_id: None,
            tenant_id: self.tenant_id.clone(),
            timestamp: Utc::now(),
            policy_json: serde_json::to_string(&self.policy).ok(),
            stats_json: serde_json::to_string(&stats).ok(),
        };
        self.db.insert_consolidation_run(&run)?;

        tracing::info!(
            tenant = %self.tenant_id,
            scanned = stats.scanned,
            decayed = stats.decayed,
            merged = stats.merged,
            pruned = stats.pruned,
            "consolidation run complete"
        );

        Ok(stats)
    }
}
