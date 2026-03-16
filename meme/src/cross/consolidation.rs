//! Memory consolidation worker — decay, merge, and prune old entries.
//!
//! Periodically maintains memory quality through three phases:
//! 1. **Decay** — reduce importance of old entries over time
//! 2. **Merge** — combine near-duplicate entries with high semantic similarity
//! 3. **Prune** — soft-delete entries whose importance has fallen below a threshold

use std::collections::HashSet;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::config::CrossConfig;
use crate::error::Result;
use crate::model::{ConsolidationRun, CrossEntry};
use crate::store::SqliteStore;

/// Configurable parameters for a single consolidation pass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    pub const fn from_config(cfg: &CrossConfig) -> Self {
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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Number of entries whose importance was decayed.
    pub decayed: usize,
    /// Number of entries that were merged.
    pub merged: usize,
    /// Number of entries that were pruned (soft-deleted).
    pub pruned: usize,
    /// Total entries scanned.
    pub scanned: usize,
    /// Duration in seconds.
    pub duration_secs: f64,
}

/// Consolidation worker that maintains memory quality over time.
///
/// Operates on a list of `CrossEntry` objects passed in by the caller.
/// The caller is responsible for fetching entries from the vector store
/// and persisting the changes (importance updates, superseded marks) back.
#[derive(Clone, Copy)]
pub struct ConsolidationWorker {
    policy: ConsolidationPolicy,
}

impl std::fmt::Debug for ConsolidationWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsolidationWorker")
            .field("policy", &self.policy)
            .finish()
    }
}

/// Actions computed by consolidation that the caller must persist.
#[derive(Debug, Default)]
pub struct ConsolidationActions {
    /// Entries whose importance should be updated: `(entry_id_str, new_importance)`.
    pub importance_updates: Vec<(String, f64)>,
    /// Entries that should be marked as superseded: `(loser_id_str, winner_id_str)`.
    pub superseded: Vec<(String, String)>,
    /// Entries that should be pruned (marked superseded by `"__pruned__"`).
    pub pruned: Vec<String>,
    /// Statistics.
    pub stats: ConsolidationStats,
}

impl ConsolidationWorker {
    /// Create a new consolidation worker.
    #[must_use]
    pub const fn new(policy: ConsolidationPolicy) -> Self {
        Self { policy }
    }

    /// Compute consolidation actions for a set of cross-session entries.
    ///
    /// This is a pure computation — it does not write to any store.
    /// The caller must apply the returned [`ConsolidationActions`] to the vector store.
    ///
    /// `vectors` must be parallel to `entries` — `vectors[i]` is the embedding for `entries[i]`.
    #[must_use]
    pub fn compute(
        &self,
        entries: &mut [CrossEntry],
        vectors: &[Vec<f32>],
    ) -> ConsolidationActions {
        let t0 = std::time::Instant::now();
        let scanned = entries.len();

        let decayed = self.decay_old_entries(entries);
        let (superseded, merged) = self.merge_similar_entries(entries, vectors);
        let (pruned_ids, pruned) = self.prune_low_importance(entries);

        let duration_secs = t0.elapsed().as_secs_f64();

        let importance_updates: Vec<(String, f64)> = entries
            .iter()
            .filter(|e| e.superseded_by.is_none())
            .map(|e| (e.entry.id.to_string(), e.importance))
            .collect();

        tracing::info!(
            scanned,
            decayed,
            merged,
            pruned,
            duration_secs,
            "consolidation computed"
        );

        ConsolidationActions {
            importance_updates,
            superseded,
            pruned: pruned_ids,
            stats: ConsolidationStats {
                decayed,
                merged,
                pruned,
                scanned,
                duration_secs,
            },
        }
    }

    /// Record a consolidation run in `SQLite`.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn record_run(
        db: &SqliteStore,
        tenant_id: &str,
        policy: &ConsolidationPolicy,
        stats: &ConsolidationStats,
    ) -> Result<()> {
        let run = ConsolidationRun {
            run_id: None,
            tenant_id: tenant_id.to_owned(),
            timestamp: Utc::now(),
            policy_json: serde_json::to_string(policy).ok(),
            stats_json: serde_json::to_string(stats).ok(),
        };
        db.insert_consolidation_run(&run)?;
        Ok(())
    }

    fn decay_old_entries(&self, entries: &mut [CrossEntry]) -> usize {
        let now = Utc::now();
        let max_age_secs = f64::from(self.policy.max_age_days) * 86400.0;
        let mut decayed = 0;

        for entry in entries.iter_mut() {
            if entry.superseded_by.is_some() {
                continue;
            }
            let Some(valid_from) = entry.valid_from else {
                continue;
            };
            let age_secs = (now - valid_from).num_seconds() as f64;
            if age_secs <= max_age_secs {
                continue;
            }

            let new_importance = entry.importance * self.policy.decay_factor;
            tracing::debug!(
                entry_id = %entry.entry.id,
                old = entry.importance,
                new = new_importance,
                age_days = age_secs / 86400.0,
                "decayed entry"
            );
            entry.importance = new_importance;
            decayed += 1;
        }

        decayed
    }

    fn merge_similar_entries(
        &self,
        entries: &mut [CrossEntry],
        vectors: &[Vec<f32>],
    ) -> (Vec<(String, String)>, usize) {
        let n = entries.len();
        if n < 2 || vectors.len() != n {
            return (Vec::new(), 0);
        }

        let mut merged_ids: HashSet<String> = HashSet::new();
        let mut superseded = Vec::new();
        let mut merged_count = 0;

        for i in 0..n {
            if entries[i].superseded_by.is_some()
                || merged_ids.contains(&entries[i].entry.id.to_string())
            {
                continue;
            }
            for j in (i + 1)..n {
                if entries[j].superseded_by.is_some()
                    || merged_ids.contains(&entries[j].entry.id.to_string())
                {
                    continue;
                }

                let sim = cosine_similarity(&vectors[i], &vectors[j]);
                if sim < self.policy.merge_similarity_threshold {
                    continue;
                }

                let (winner_idx, loser_idx) = if entries[i].importance >= entries[j].importance {
                    (i, j)
                } else {
                    (j, i)
                };

                let winner_id = entries[winner_idx].entry.id;
                let loser_id = entries[loser_idx].entry.id;

                entries[loser_idx].superseded_by = Some(winner_id);
                merged_ids.insert(loser_id.to_string());
                superseded.push((loser_id.to_string(), winner_id.to_string()));
                merged_count += 1;

                tracing::debug!(
                    loser = %loser_id,
                    winner = %winner_id,
                    similarity = sim,
                    "merged entries"
                );
            }
        }

        (superseded, merged_count)
    }

    fn prune_low_importance(&self, entries: &mut [CrossEntry]) -> (Vec<String>, usize) {
        let mut pruned_ids = Vec::new();

        for entry in entries.iter_mut() {
            if entry.superseded_by.is_some() {
                continue;
            }
            if entry.importance >= self.policy.min_importance {
                continue;
            }

            tracing::debug!(
                entry_id = %entry.entry.id,
                importance = entry.importance,
                "pruned entry"
            );
            pruned_ids.push(entry.entry.id.to_string());
        }

        let count = pruned_ids.len();
        (pruned_ids, count)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let mag_a: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let mag_b: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}
