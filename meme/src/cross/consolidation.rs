//! Memory consolidation worker — decay, merge, and prune old entries.
//!
//! Periodically maintains memory quality through three phases:
//! 1. **Decay** — reduce importance of old entries over time
//! 2. **Merge** — combine near-duplicate entries with high semantic similarity
//! 3. **Prune** — soft-delete entries whose importance has fallen below a threshold

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::config::CrossConfig;
use crate::model::MemoryEntry;

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
/// Pure computation on `MemoryEntry` slices — the caller fetches entries
/// from the vector store and applies the returned deletions.
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
    /// Entries that should be marked as superseded: `(loser_id, winner_id)`.
    pub superseded: Vec<(String, String)>,
    /// Entry IDs that should be deleted.
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

    /// Compute consolidation actions for a set of memory entries.
    ///
    /// Pure computation — does not write to any store.
    /// `vectors` must be parallel to `entries`.
    #[must_use]
    pub fn compute(
        &self,
        entries: &mut [MemoryEntry],
        vectors: &[Vec<f32>],
    ) -> ConsolidationActions {
        let t0 = std::time::Instant::now();
        let scanned = entries.len();
        let mut importance: Vec<f64> = vec![1.0; scanned];
        let mut dead: Vec<bool> = vec![false; scanned];

        let decayed = self.decay(entries, &mut importance, &mut dead);
        let (superseded, merged) = self.merge(entries, vectors, &importance, &mut dead);
        let (pruned_ids, pruned) = self.prune(entries, &importance, &dead);

        let duration_secs = t0.elapsed().as_secs_f64();
        tracing::info!(
            scanned,
            decayed,
            merged,
            pruned,
            duration_secs,
            "consolidation computed"
        );

        ConsolidationActions {
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

    fn decay(&self, entries: &[MemoryEntry], importance: &mut [f64], dead: &mut [bool]) -> usize {
        let now = chrono::Utc::now();
        let max_age_secs = f64::from(self.policy.max_age_days) * 86400.0;
        let mut count = 0;
        for (i, entry) in entries.iter().enumerate() {
            let Some(ts) = entry.timestamp else { continue };
            let age = (now - ts).num_seconds() as f64;
            if age > max_age_secs {
                importance[i] *= self.policy.decay_factor;
                if importance[i] < self.policy.min_importance {
                    dead[i] = true;
                }
                count += 1;
            }
        }
        count
    }

    fn merge(
        &self,
        entries: &[MemoryEntry],
        vectors: &[Vec<f32>],
        importance: &[f64],
        dead: &mut [bool],
    ) -> (Vec<(String, String)>, usize) {
        let n = entries.len();
        if n < 2 || vectors.len() != n {
            return (Vec::new(), 0);
        }
        let mut merged_ids: HashSet<usize> = HashSet::new();
        let mut superseded = Vec::new();
        let mut count = 0;

        for i in 0..n {
            if dead[i] || merged_ids.contains(&i) {
                continue;
            }
            for j in (i + 1)..n {
                if dead[j] || merged_ids.contains(&j) {
                    continue;
                }
                let sim = cosine_similarity(&vectors[i], &vectors[j]);
                if sim < self.policy.merge_similarity_threshold {
                    continue;
                }

                let (loser, winner) = if importance[i] >= importance[j] {
                    (j, i)
                } else {
                    (i, j)
                };
                dead[loser] = true;
                merged_ids.insert(loser);
                superseded.push((
                    entries[loser].id.to_string(),
                    entries[winner].id.to_string(),
                ));
                count += 1;
            }
        }
        (superseded, count)
    }

    fn prune(
        &self,
        entries: &[MemoryEntry],
        importance: &[f64],
        dead: &[bool],
    ) -> (Vec<String>, usize) {
        let mut ids = Vec::new();
        for (i, entry) in entries.iter().enumerate() {
            if !dead[i] && importance[i] < self.policy.min_importance {
                ids.push(entry.id.to_string());
            }
        }
        let count = ids.len();
        (ids, count)
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
