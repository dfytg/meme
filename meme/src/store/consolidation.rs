//! Memory consolidation — decay, merge near-duplicates, prune low-importance entries.

use std::collections::{HashMap, HashSet};

use super::VectorStore;
use crate::error::Result;
use crate::model::Memory;

/// Parameters for a consolidation run.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationParams {
    /// Maximum age in days before decay applies.
    pub max_age_days: u32,
    /// Decay factor (0.0–1.0) applied to old entries' importance.
    pub decay_factor: f64,
    /// Cosine similarity threshold for merging near-duplicates.
    pub merge_threshold: f64,
    /// Minimum importance score to keep an entry.
    pub min_importance: f64,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        Self {
            max_age_days: 90,
            decay_factor: 0.95,
            merge_threshold: 0.95,
            min_importance: 0.1,
        }
    }
}

/// Statistics from a consolidation run.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationStats {
    /// Total entries scanned.
    pub scanned: usize,
    /// Entries whose importance was decayed.
    pub decayed: usize,
    /// Entries merged (near-duplicates removed).
    pub merged: usize,
    /// Entries pruned (below importance threshold).
    pub pruned: usize,
    /// Duration in seconds.
    pub duration_secs: f64,
}

/// Consolidate memory: decay old entries, merge near-duplicates, prune low-importance.
///
/// Operates within the given namespace to respect multi-tenant isolation.
/// Uses ANN search per entry to find near-duplicates (O(n·k) instead of O(n²)).
///
/// # Errors
///
/// Returns an error if reading or deleting entries fails.
pub async fn consolidate(
    store: &VectorStore,
    params: &ConsolidationParams,
    namespace: Option<&str>,
) -> Result<ConsolidationStats> {
    let pairs = store.get_all_with_vectors(namespace).await?;
    if pairs.is_empty() {
        return Ok(ConsolidationStats::default());
    }

    let t0 = std::time::Instant::now();
    let (entries, vectors): (Vec<Memory>, Vec<Vec<f32>>) = pairs.into_iter().unzip();
    let n = entries.len();

    let now = chrono::Utc::now();
    let max_age_secs = f64::from(params.max_age_days) * 86400.0;
    let mut importance: Vec<f64> = vec![1.0; n];
    let mut dead = HashSet::new();

    let mut decayed = 0usize;
    for (i, entry) in entries.iter().enumerate() {
        let Some(ts) = entry.timestamp else { continue };
        let age = (now - ts).num_seconds() as f64;
        if age > max_age_secs {
            importance[i] *= params.decay_factor;
            if importance[i] < params.min_importance {
                dead.insert(i);
            }
            decayed += 1;
        }
    }

    let merge_k = 5;
    let mut merged = 0usize;
    let mut ids_to_delete: Vec<uuid::Uuid> = Vec::new();

    let id_to_idx: HashMap<uuid::Uuid, usize> =
        entries.iter().enumerate().map(|(i, e)| (e.id, i)).collect();

    let live_indices: Vec<usize> = (0..n).filter(|i| !dead.contains(i)).collect();
    let max_ann_workers = 8;
    let semaphore = tokio::sync::Semaphore::new(max_ann_workers);
    let vectors_ref = &vectors;
    let ann_futures = live_indices.iter().map(|&i| {
        let sem = &semaphore;
        async move {
            let _permit = sem.acquire().await;
            store
                .semantic_search(&vectors_ref[i], merge_k, namespace)
                .await
                .map(|neighbors| (i, neighbors))
        }
    });
    let all_neighbors: Vec<(usize, Vec<Memory>)> =
        futures::future::try_join_all(ann_futures).await?;

    for (i, neighbors) in all_neighbors {
        if dead.contains(&i) {
            continue;
        }
        for neighbor in &neighbors {
            if neighbor.id == entries[i].id {
                continue;
            }
            let Some(&j) = id_to_idx.get(&neighbor.id) else {
                continue;
            };
            if dead.contains(&j) {
                continue;
            }
            let sim = cosine_similarity(&vectors[i], &vectors[j]);
            if sim >= params.merge_threshold {
                let loser = if importance[i] >= importance[j] { j } else { i };
                if dead.insert(loser) {
                    ids_to_delete.push(entries[loser].id);
                    merged += 1;
                }
            }
        }
    }

    let mut pruned = 0usize;
    for (i, entry) in entries.iter().enumerate() {
        if !dead.contains(&i) && importance[i] < params.min_importance {
            ids_to_delete.push(entry.id);
            pruned += 1;
        }
    }

    if !ids_to_delete.is_empty() {
        store.delete_entries(&ids_to_delete).await?;
    }

    let stats = ConsolidationStats {
        scanned: n,
        decayed,
        merged,
        pruned,
        duration_secs: t0.elapsed().as_secs_f64(),
    };
    tracing::info!(?stats, "consolidation complete");
    Ok(stats)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let mag_a: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_arbitrary() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = 3.0f64.mul_add(6.0, 2.0f64.mul_add(5.0, 1.0 * 4.0));
        let mag_a = (1.0_f64 + 4.0 + 9.0).sqrt();
        let mag_b = (16.0_f64 + 25.0 + 36.0).sqrt();
        let expected = dot / (mag_a * mag_b);
        assert!((cosine_similarity(&a, &b) - expected).abs() < 1e-9);
    }
}
