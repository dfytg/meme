//! Vector store — multi-view indexing with `LanceDB`.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::error::{Error, Result};
use crate::model::{MemoryEntry, MetadataFilter, Scope};

/// Build a SQL WHERE clause fragment for a [`Scope`].
/// Returns `None` if no scope is set.
fn scope_to_where_clause(scope: &Scope) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(uid) = &scope.user_id {
        parts.push(format!("user_id = '{}'", escape_sql_string(uid)));
    }
    if let Some(sid) = &scope.session_id {
        parts.push(format!("session_id = '{}'", escape_sql_string(sid)));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

/// `LanceDB`-backed vector store with multi-view indexing.
pub struct VectorStore {
    db: lancedb::Connection,
    table_name: String,
    dimension: usize,
    cached_table: tokio::sync::RwLock<Option<lancedb::Table>>,
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("table_name", &self.table_name)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl VectorStore {
    /// Open or create a `LanceDB` store.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn open(db_path: &str, table_name: &str, dimension: usize) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;
        let db = lancedb::connect(db_path).execute().await?;

        let store = Self {
            db,
            table_name: table_name.to_owned(),
            dimension,
            cached_table: tokio::sync::RwLock::new(None),
        };
        store.ensure_table().await?;
        Ok(store)
    }

    async fn ensure_table(&self) -> Result<()> {
        let tables = self.db.table_names().execute().await?;

        if tables.contains(&self.table_name) {
            tracing::info!(table = %self.table_name, "opened existing LanceDB table");
            let table = self.get_table().await?;
            self.rebuild_fts_index(&table).await;
        } else {
            let schema = self.build_schema();
            self.db
                .create_empty_table(&self.table_name, schema)
                .execute()
                .await?;
            tracing::info!(table = %self.table_name, "created new LanceDB table");
        }
        Ok(())
    }

    fn build_schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("entry_id", DataType::Utf8, false),
            Field::new("restatement", DataType::Utf8, false),
            Field::new("keywords_text", DataType::Utf8, false),
            Field::new("timestamp", DataType::Utf8, true),
            Field::new("location", DataType::Utf8, true),
            Field::new("persons_text", DataType::Utf8, false),
            Field::new("entities_text", DataType::Utf8, false),
            Field::new("topic", DataType::Utf8, true),
            Field::new("user_id", DataType::Utf8, true),
            Field::new("session_id", DataType::Utf8, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    {
                        self.dimension as i32
                    },
                ),
                false,
            ),
        ]))
    }

    async fn get_table(&self) -> Result<lancedb::Table> {
        {
            let guard = self.cached_table.read().await;
            if let Some(table) = guard.as_ref() {
                return Ok(table.clone());
            }
        }
        let table = self.db.open_table(&self.table_name).execute().await?;
        *self.cached_table.write().await = Some(table.clone());
        Ok(table)
    }

    async fn invalidate_cache(&self) {
        *self.cached_table.write().await = None;
    }

    /// Rebuild the FTS index to include all current data.
    ///
    /// `LanceDB` FTS is snapshot-based — new rows are invisible to keyword search
    /// until the index is recreated.
    async fn rebuild_fts_index(&self, table: &lancedb::Table) {
        if let Err(e) = table
            .create_index(
                &["restatement"],
                lancedb::index::Index::FTS(lancedb::index::scalar::FtsIndexBuilder::default()),
            )
            .execute()
            .await
        {
            tracing::debug!(error = %e, "FTS index rebuild skipped");
        }
    }

    fn batch_to_entries(batch: &RecordBatch) -> Vec<MemoryEntry> {
        let col = |name| -> Option<&StringArray> {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        };
        let id_col = col("entry_id");
        let rest_col = col("restatement");
        let kw_col = col("keywords_text");
        let ts_col = col("timestamp");
        let loc_col = col("location");
        let per_col = col("persons_text");
        let ent_col = col("entities_text");
        let topic_col = col("topic");
        let uid_col = col("user_id");
        let sid_col = col("session_id");

        (0..batch.num_rows())
            .map(|i| {
                let id_str = str_val(id_col, i);
                MemoryEntry {
                    id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                    restatement: str_val(rest_col, i),
                    keywords: str_val(kw_col, i)
                        .split("||")
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    timestamp: opt_val(ts_col, i).and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                    }),
                    location: opt_val(loc_col, i),
                    persons: str_val(per_col, i)
                        .split("||")
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    entities: str_val(ent_col, i)
                        .split("||")
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    topic: opt_val(topic_col, i),
                    user_id: opt_val(uid_col, i),
                    session_id: opt_val(sid_col, i),
                }
            })
            .collect()
    }

    /// Add entries with their pre-computed embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the table cannot be opened or data insertion fails.
    pub async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        if entries.len() != vectors.len() {
            return Err(Error::VectorStore(format!(
                "entries/vectors length mismatch: {} vs {}",
                entries.len(),
                vectors.len()
            )));
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dimension {
                return Err(Error::VectorStore(format!(
                    "vector[{i}] dimension mismatch: expected {}, got {}",
                    self.dimension,
                    v.len()
                )));
            }
        }

        let n = entries.len();
        let schema = self.build_schema();
        let col = |f: fn(&MemoryEntry) -> String| -> ArrayRef {
            Arc::new(StringArray::from(entries.iter().map(f).collect::<Vec<_>>()))
        };

        let columns: Vec<ArrayRef> = vec![
            col(|e| e.id.to_string()),
            col(|e| e.restatement.clone()),
            col(|e| e.keywords.join("||")),
            col(|e| e.timestamp.map(|ts| ts.to_rfc3339()).unwrap_or_default()),
            col(|e| e.location.clone().unwrap_or_default()),
            col(|e| e.persons.join("||")),
            col(|e| e.entities.join("||")),
            col(|e| e.topic.clone().unwrap_or_default()),
            col(|e| e.user_id.clone().unwrap_or_default()),
            col(|e| e.session_id.clone().unwrap_or_default()),
            build_vector_column(vectors, self.dimension)?,
        ];

        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).map_err(Error::arrow)?;
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let table = self.get_table().await?;
        table.add(reader).execute().await?;

        self.rebuild_fts_index(&table).await;

        tracing::info!(count = n, "added memory entries");
        Ok(())
    }

    /// Semantic search by vector similarity.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector query fails.
    pub async fn semantic_search(
        &self,
        query_vec: &[f32],
        top_k: usize,
        scope: &Scope,
    ) -> Result<Vec<MemoryEntry>> {
        let table = self.get_table().await?;
        if table.count_rows(None).await? == 0 {
            return Ok(Vec::new());
        }
        let mut q = table.query().nearest_to(query_vec)?;
        q = q.limit(top_k);
        if let Some(clause) = scope_to_where_clause(scope) {
            q = q.only_if(clause);
        }
        self.collect_entries(q.execute().await?).await
    }

    /// Keyword search (scans restatement text for keyword matches).
    ///
    /// # Errors
    ///
    /// Returns an error if the search query fails.
    pub async fn keyword_search(
        &self,
        keywords: &[String],
        top_k: usize,
        scope: &Scope,
    ) -> Result<Vec<MemoryEntry>> {
        if keywords.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.get_table().await?;
        if table.count_rows(None).await? == 0 {
            return Ok(Vec::new());
        }

        let scope_clause = scope_to_where_clause(scope);
        let fts_query = keywords.join(" ");

        // Try FTS first; fall back to LIKE on failure.
        if let Ok(stream) = table
            .query()
            .full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(
                fts_query.clone(),
            ))
            .limit(top_k)
            .execute()
            .await
        {
            let mut entries = self.collect_entries(stream).await?;
            if scope_clause.is_some() {
                entries.retain(|e| scope_matches(e, scope));
            }
            return Ok(entries);
        }

        let conditions: Vec<String> = keywords
            .iter()
            .map(|kw| {
                let safe = escape_like(kw);
                format!("(restatement LIKE '%{safe}%' OR keywords_text LIKE '%{safe}%')")
            })
            .collect();
        let mut where_clause = format!("({})", conditions.join(" OR "));
        if let Some(sc) = &scope_clause {
            where_clause = format!("{where_clause} AND {sc}");
        }

        let stream = table
            .query()
            .only_if(where_clause)
            .limit(top_k)
            .execute()
            .await?;
        self.collect_entries(stream).await
    }

    /// Structured search by metadata filtering.
    ///
    /// # Errors
    ///
    /// Returns an error if the filter query fails.
    pub async fn structured_search(
        &self,
        filter: &MetadataFilter,
        top_k: usize,
        scope: &Scope,
    ) -> Result<Vec<MemoryEntry>> {
        if filter.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.get_table().await?;
        if table.count_rows(None).await? == 0 {
            return Ok(Vec::new());
        }

        let mut conditions = Vec::new();
        if let Some(persons) = &filter.persons {
            let conds: Vec<String> = persons
                .iter()
                .map(|p| format!("persons_text LIKE '%{}%'", escape_like(p)))
                .collect();
            if !conds.is_empty() {
                conditions.push(format!("({})", conds.join(" OR ")));
            }
        }
        if let Some(location) = &filter.location {
            conditions.push(format!("location LIKE '%{}%'", escape_like(location)));
        }
        if let Some(entities) = &filter.entities {
            let conds: Vec<String> = entities
                .iter()
                .map(|e| format!("entities_text LIKE '%{}%'", escape_like(e)))
                .collect();
            if !conds.is_empty() {
                conditions.push(format!("({})", conds.join(" OR ")));
            }
        }
        if let Some((start, end)) = &filter.timestamp_range {
            conditions.push(format!(
                "timestamp >= '{}' AND timestamp <= '{}'",
                start.to_rfc3339(),
                end.to_rfc3339()
            ));
        }
        if conditions.is_empty() {
            return Ok(Vec::new());
        }

        let mut where_clause = conditions.join(" AND ");
        if let Some(sc) = scope_to_where_clause(scope) {
            where_clause = format!("{where_clause} AND {sc}");
        }
        let stream = table
            .query()
            .only_if(where_clause)
            .limit(top_k)
            .execute()
            .await?;
        self.collect_entries(stream).await
    }

    /// Retrieve all entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn get_all(&self, scope: &Scope) -> Result<Vec<MemoryEntry>> {
        let table = self.get_table().await?;
        let mut q = table.query();
        if let Some(clause) = scope_to_where_clause(scope) {
            q = q.only_if(clause);
        }
        self.collect_entries(q.execute().await?).await
    }

    /// Retrieve all entries together with their embedding vectors, filtered by scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn get_all_with_vectors(
        &self,
        scope: &Scope,
    ) -> Result<Vec<(MemoryEntry, Vec<f32>)>> {
        let table = self.get_table().await?;
        let mut q = table.query();
        if let Some(clause) = scope_to_where_clause(scope) {
            q = q.only_if(clause);
        }
        let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
        Ok(batches
            .iter()
            .flat_map(|b| {
                let entries = Self::batch_to_entries(b);
                let vectors = batch_to_vectors(b, self.dimension);
                entries.into_iter().zip(vectors)
            })
            .collect())
    }

    /// Retrieve a single entry by its UUID.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_by_id(&self, id: uuid::Uuid) -> Result<Option<MemoryEntry>> {
        let table = self.get_table().await?;
        let stream = table
            .query()
            .only_if(format!("entry_id = '{id}'"))
            .limit(1)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        Ok(batches.iter().flat_map(Self::batch_to_entries).next())
    }

    /// Replace an existing entry (delete + re-insert with same ID).
    ///
    /// # Errors
    ///
    /// Returns an error if the delete or insert fails.
    pub async fn update_entry(&self, entry: &MemoryEntry, vector: &[f32]) -> Result<()> {
        self.delete_entries(&[entry.id.to_string()]).await?;
        self.add_entries(std::slice::from_ref(entry), &[vector.to_vec()])
            .await
    }

    /// Delete entries by their IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete operation fails.
    pub async fn delete_entries(&self, entry_ids: &[String]) -> Result<usize> {
        if entry_ids.is_empty() {
            return Ok(0);
        }
        for id in entry_ids {
            if !is_valid_uuid(id) {
                return Err(Error::VectorStore(format!("invalid entry id: {id}")));
            }
        }
        let table = self.get_table().await?;
        let ids_csv: String = entry_ids
            .iter()
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(", ");
        table.delete(&format!("entry_id IN ({ids_csv})")).await?;
        let count = entry_ids.len();
        tracing::info!(count, "deleted entries from vector store");
        Ok(count)
    }

    /// Count the total number of entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn count(&self, scope: &Scope) -> Result<usize> {
        let table = self.get_table().await?;
        Ok(table.count_rows(scope_to_where_clause(scope)).await?)
    }

    async fn collect_entries(
        &self,
        stream: impl futures::Stream<Item = std::result::Result<RecordBatch, lancedb::Error>>
        + Send
        + Unpin,
    ) -> Result<Vec<MemoryEntry>> {
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        Ok(batches.iter().flat_map(Self::batch_to_entries).collect())
    }

    /// Clear entries matching the given scope, or all entries if scope is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete operation fails.
    pub async fn clear(&self, scope: &Scope) -> Result<()> {
        if let Some(clause) = scope_to_where_clause(scope) {
            let table = self.get_table().await?;
            table.delete(&clause).await?;
            tracing::info!(table = %self.table_name, %clause, "cleared scoped entries");
        } else {
            self.clear_all().await?;
        }
        Ok(())
    }

    /// Drop and recreate the entire table. **Removes all tenants' data.**
    ///
    /// # Errors
    ///
    /// Returns an error if the table cannot be dropped or recreated.
    pub async fn clear_all(&self) -> Result<()> {
        self.invalidate_cache().await;
        self.db.drop_table(&self.table_name, &[]).await?;
        self.ensure_table().await?;
        tracing::info!(table = %self.table_name, "cleared entire vector store");
        Ok(())
    }

    /// Consolidate memory: decay old entries, merge near-duplicates, prune low-importance.
    ///
    /// Operates within the given `scope` to respect multi-tenant isolation.
    /// Uses ANN search per entry to find near-duplicates (O(n·k) instead of O(n²)).
    ///
    /// # Errors
    ///
    /// Returns an error if reading or deleting entries fails.
    pub async fn consolidate(
        &self,
        max_age_days: u32,
        decay_factor: f64,
        merge_threshold: f64,
        min_importance: f64,
        scope: &Scope,
    ) -> Result<ConsolidationStats> {
        let pairs = self.get_all_with_vectors(scope).await?;
        if pairs.is_empty() {
            return Ok(ConsolidationStats::default());
        }

        let t0 = std::time::Instant::now();
        let (entries, vectors): (Vec<MemoryEntry>, Vec<Vec<f32>>) = pairs.into_iter().unzip();
        let n = entries.len();

        let now = chrono::Utc::now();
        let max_age_secs = f64::from(max_age_days) * 86400.0;
        let mut importance: Vec<f64> = vec![1.0; n];
        let mut dead = std::collections::HashSet::new();

        let mut decayed = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            let Some(ts) = entry.timestamp else { continue };
            let age = (now - ts).num_seconds() as f64;
            if age > max_age_secs {
                importance[i] *= decay_factor;
                if importance[i] < min_importance {
                    dead.insert(i);
                }
                decayed += 1;
            }
        }

        // ANN-based near-duplicate detection: batch parallel neighbor queries.
        let merge_k = 5;
        let mut merged = 0usize;
        let mut ids_to_delete: Vec<String> = Vec::new();

        // Build a lookup from entry ID → index for O(1) neighbor resolution.
        let id_to_idx: std::collections::HashMap<uuid::Uuid, usize> =
            entries.iter().enumerate().map(|(i, e)| (e.id, i)).collect();

        // Parallel ANN queries for all live entries.
        let live_indices: Vec<usize> = (0..n).filter(|i| !dead.contains(i)).collect();
        let max_ann_workers = 8;
        let semaphore = tokio::sync::Semaphore::new(max_ann_workers);
        let vectors_ref = &vectors;
        let ann_futures = live_indices.iter().map(|&i| {
            let sem = &semaphore;
            async move {
                let _permit = sem.acquire().await;
                self.semantic_search(&vectors_ref[i], merge_k, scope)
                    .await
                    .map(|neighbors| (i, neighbors))
            }
        });
        let all_neighbors: Vec<(usize, Vec<MemoryEntry>)> =
            futures::future::try_join_all(ann_futures).await?;

        // Process neighbors sequentially (mutates `dead`).
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
                if sim >= merge_threshold {
                    let loser = if importance[i] >= importance[j] { j } else { i };
                    if dead.insert(loser) {
                        ids_to_delete.push(entries[loser].id.to_string());
                        merged += 1;
                    }
                }
            }
        }

        let mut pruned = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            if !dead.contains(&i) && importance[i] < min_importance {
                ids_to_delete.push(entry.id.to_string());
                pruned += 1;
            }
        }

        if !ids_to_delete.is_empty() {
            self.delete_entries(&ids_to_delete).await?;
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

fn scope_matches(entry: &MemoryEntry, scope: &Scope) -> bool {
    if let Some(uid) = &scope.user_id
        && entry.user_id.as_deref() != Some(uid.as_str())
    {
        return false;
    }
    if let Some(sid) = &scope.session_id
        && entry.session_id.as_deref() != Some(sid.as_str())
    {
        return false;
    }
    true
}

fn str_val(col: Option<&StringArray>, i: usize) -> String {
    col.map(|c| c.value(i).to_owned()).unwrap_or_default()
}

fn opt_val(col: Option<&StringArray>, i: usize) -> Option<String> {
    col.filter(|c| !c.is_null(i))
        .map(|c| c.value(i))
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn build_vector_column(vectors: &[Vec<f32>], dimension: usize) -> Result<ArrayRef> {
    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let values = Float32Array::from(flat);
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let array = FixedSizeListArray::try_new(field, dimension as i32, Arc::new(values), None)
        .map_err(Error::arrow)?;
    Ok(Arc::new(array))
}

/// Escape a string for use in SQL `LIKE` patterns.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "''")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Escape a string for use in SQL equality (`=`) comparisons.
/// Only escapes single quotes to prevent SQL injection.
fn escape_sql_string(s: &str) -> String {
    s.replace('\'', "''")
}

fn is_valid_uuid(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok()
}

fn batch_to_vectors(batch: &RecordBatch, dimension: usize) -> Vec<Vec<f32>> {
    let n = batch.num_rows();
    let Some(col) = batch.column_by_name("vector") else {
        return vec![Vec::new(); n];
    };
    let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() else {
        return vec![Vec::new(); n];
    };
    let values = fsl.values();
    let Some(float_values) = values.as_any().downcast_ref::<Float32Array>() else {
        return vec![Vec::new(); n];
    };

    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dimension;
        let end = start + dimension;
        if end <= float_values.len() {
            vectors.push(float_values.values()[start..end].to_vec());
        } else {
            vectors.push(Vec::new());
        }
    }
    vectors
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
    fn escape_like_special_chars() {
        assert_eq!(escape_like("it's"), "it''s");
        assert_eq!(escape_like("100%"), "100\\%");
        assert_eq!(escape_like("a_b"), "a\\_b");
    }

    #[test]
    fn escape_like_clean_string() {
        assert_eq!(escape_like("hello world"), "hello world");
    }

    #[test]
    fn escape_like_combined() {
        assert_eq!(escape_like("it's 100%_done"), "it''s 100\\%\\_done");
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-9);
        assert!((cosine_similarity(&b, &a) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_arbitrary() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = 3.0f64.mul_add(6.0, 2.0f64.mul_add(5.0, 1.0 * 4.0));
        let mag_a = (1.0_f64 + 4.0 + 9.0).sqrt();
        let mag_b = (16.0_f64 + 25.0 + 36.0).sqrt();
        let expected = dot / (mag_a * mag_b);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - expected).abs() < 1e-9);
    }

    #[test]
    fn escape_like_backslash() {
        assert_eq!(escape_like(r"a\b"), r"a\\b");
        assert_eq!(escape_like(r"c:\path"), r"c:\\path");
    }

    #[test]
    fn is_valid_uuid_valid() {
        let id = uuid::Uuid::new_v4().to_string();
        assert!(is_valid_uuid(&id));
    }

    #[test]
    fn is_valid_uuid_invalid() {
        assert!(!is_valid_uuid("not-a-uuid"));
        assert!(!is_valid_uuid(""));
        assert!(!is_valid_uuid("'; DROP TABLE --"));
    }

    #[test]
    fn scope_empty_no_clause() {
        let s = Scope::default();
        assert!(scope_to_where_clause(&s).is_none());
    }

    #[test]
    fn scope_user_only() {
        let s = Scope {
            user_id: Some("alice".into()),
            session_id: None,
        };
        let clause = scope_to_where_clause(&s).unwrap();
        assert!(clause.contains("user_id"));
        assert!(clause.contains("alice"));
        assert!(!clause.contains("session_id"));
    }

    #[test]
    fn scope_both() {
        let s = Scope {
            user_id: Some("bob".into()),
            session_id: Some("s1".into()),
        };
        let clause = scope_to_where_clause(&s).unwrap();
        assert!(clause.contains("user_id"));
        assert!(clause.contains("session_id"));
        assert!(clause.contains("AND"));
    }

    #[test]
    fn scope_matches_no_scope() {
        let e = MemoryEntry::new("test");
        let s = Scope::default();
        assert!(scope_matches(&e, &s));
    }

    #[test]
    fn scope_matches_user_hit() {
        let mut e = MemoryEntry::new("test");
        e.user_id = Some("alice".into());
        let s = Scope {
            user_id: Some("alice".into()),
            session_id: None,
        };
        assert!(scope_matches(&e, &s));
    }

    #[test]
    fn scope_matches_user_miss() {
        let mut e = MemoryEntry::new("test");
        e.user_id = Some("bob".into());
        let s = Scope {
            user_id: Some("alice".into()),
            session_id: None,
        };
        assert!(!scope_matches(&e, &s));
    }

    #[test]
    fn escape_sql_string_quotes() {
        assert_eq!(escape_sql_string("it's"), "it''s");
        assert_eq!(escape_sql_string("hello"), "hello");
    }

    #[test]
    fn escape_sql_string_preserves_underscores() {
        assert_eq!(escape_sql_string("user_a"), "user_a");
        assert_eq!(escape_sql_string("a_b_c"), "a_b_c");
    }

    #[test]
    fn scope_with_underscore_no_escape() {
        let s = Scope {
            user_id: Some("user_a".into()),
            session_id: None,
        };
        let clause = scope_to_where_clause(&s).unwrap();
        assert_eq!(clause, "user_id = 'user_a'");
    }
}
