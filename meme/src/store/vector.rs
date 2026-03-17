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
use crate::model::{MemoryEntry, MetadataFilter};

/// `LanceDB`-backed vector store with multi-view indexing.
pub struct VectorStore {
    db: lancedb::Connection,
    table_name: String,
    dimension: usize,
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
        let db = lancedb::connect(db_path)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("failed to connect: {e}")))?;

        let store = Self {
            db,
            table_name: table_name.to_owned(),
            dimension,
        };
        store.ensure_table().await?;
        Ok(store)
    }

    async fn ensure_table(&self) -> Result<()> {
        let tables = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("list tables failed: {e}")))?;

        if tables.contains(&self.table_name) {
            tracing::info!(table = %self.table_name, "opened existing LanceDB table");
        } else {
            let schema = self.build_schema();
            self.db
                .create_empty_table(&self.table_name, schema)
                .execute()
                .await
                .map_err(|e| Error::VectorStore(format!("create table failed: {e}")))?;
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
        self.db
            .open_table(&self.table_name)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("open table failed: {e}")))
    }

    fn batch_to_entries(batch: &RecordBatch) -> Vec<MemoryEntry> {
        let n = batch.num_rows();
        let mut entries = Vec::with_capacity(n);

        let get_str = |name: &str| -> Option<&StringArray> {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        };

        let id_col = get_str("entry_id");
        let restatement_col = get_str("restatement");
        let keywords_col = get_str("keywords_text");
        let ts_col = get_str("timestamp");
        let loc_col = get_str("location");
        let persons_col = get_str("persons_text");
        let entities_col = get_str("entities_text");
        let topic_col = get_str("topic");

        for i in 0..n {
            let id_str = id_col.map(|c| c.value(i)).unwrap_or_default();
            let restatement = restatement_col
                .map(|c| c.value(i).to_owned())
                .unwrap_or_default();

            let keywords = keywords_col
                .map(|c| split_delimited(c.value(i)))
                .unwrap_or_default();
            let persons = persons_col
                .map(|c| split_delimited(c.value(i)))
                .unwrap_or_default();
            let entity_list = entities_col
                .map(|c| split_delimited(c.value(i)))
                .unwrap_or_default();

            let timestamp = ts_col
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i))
                .filter(|s| !s.is_empty())
                .and_then(|s| {
                    chrono::DateTime::parse_from_rfc3339(s)
                        .ok()
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                });

            let location = loc_col
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i))
                .filter(|s| !s.is_empty())
                .map(String::from);

            let topic = topic_col
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i))
                .filter(|s| !s.is_empty())
                .map(String::from);

            let id = uuid::Uuid::parse_str(id_str).unwrap_or_else(|_| uuid::Uuid::new_v4());

            entries.push(MemoryEntry {
                id,
                restatement,
                keywords,
                timestamp,
                location,
                persons,
                entities: entity_list,
                topic,
            });
        }
        entries
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

        let n = entries.len();
        let schema = self.build_schema();

        let entry_ids: ArrayRef = Arc::new(StringArray::from(
            entries.iter().map(|e| e.id.to_string()).collect::<Vec<_>>(),
        ));
        let restatements: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| e.restatement.clone())
                .collect::<Vec<_>>(),
        ));
        let keywords_text: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| join_delimited(&e.keywords))
                .collect::<Vec<_>>(),
        ));
        let timestamps: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| e.timestamp.map(|ts| ts.to_rfc3339()).unwrap_or_default())
                .collect::<Vec<_>>(),
        ));
        let locations: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| e.location.clone().unwrap_or_default())
                .collect::<Vec<_>>(),
        ));
        let persons_text: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| join_delimited(&e.persons))
                .collect::<Vec<_>>(),
        ));
        let entities_text: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| join_delimited(&e.entities))
                .collect::<Vec<_>>(),
        ));
        let topics: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| e.topic.clone().unwrap_or_default())
                .collect::<Vec<_>>(),
        ));

        let dim = self.dimension;
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let values = Float32Array::from(flat);
        let fsl_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array: ArrayRef = Arc::new(
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            FixedSizeListArray::try_new(fsl_field, dim as i32, Arc::new(values), None)
                .map_err(|e| Error::VectorStore(format!("vector array error: {e}")))?,
        );

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                entry_ids,
                restatements,
                keywords_text,
                timestamps,
                locations,
                persons_text,
                entities_text,
                topics,
                vector_array,
            ],
        )
        .map_err(|e| Error::VectorStore(format!("record batch error: {e}")))?;

        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let table = self.get_table().await?;
        let was_empty = table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?
            == 0;

        table
            .add(reader)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("add entries failed: {e}")))?;

        // Create FTS index after first data insertion.
        if was_empty {
            if let Err(e) = table
                .create_index(
                    &["restatement"],
                    lancedb::index::Index::FTS(lancedb::index::scalar::FtsIndexBuilder::default()),
                )
                .execute()
                .await
            {
                tracing::warn!(error = %e, "FTS index creation skipped");
            } else {
                tracing::info!("FTS index created on restatement column");
            }
        }

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
    ) -> Result<Vec<MemoryEntry>> {
        let table = self.get_table().await?;
        if table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?
            == 0
        {
            return Ok(Vec::new());
        }

        let results = table
            .query()
            .nearest_to(query_vec)
            .map_err(|e| Error::VectorStore(format!("nearest_to failed: {e}")))?
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("vector search failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| Self::batch_to_entries(b))
            .collect())
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
    ) -> Result<Vec<MemoryEntry>> {
        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let table = self.get_table().await?;
        if table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?
            == 0
        {
            return Ok(Vec::new());
        }

        let fts_query = keywords.join(" ");

        // Try FTS first (uses Tantivy index on restatement column).
        match table
            .query()
            .full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(
                fts_query.clone(),
            ))
            .limit(top_k)
            .execute()
            .await
        {
            Ok(stream) => {
                let batches: Vec<RecordBatch> = stream
                    .try_collect()
                    .await
                    .map_err(|e| Error::VectorStore(format!("FTS collect failed: {e}")))?;
                return Ok(batches
                    .iter()
                    .flat_map(|b| Self::batch_to_entries(b))
                    .collect());
            }
            Err(e) => {
                tracing::debug!(error = %e, "FTS search unavailable, falling back to LIKE");
            }
        }

        // Fallback to LIKE pattern matching.
        let conditions: Vec<String> = keywords
            .iter()
            .map(|kw| {
                let safe = escape_like(kw);
                format!("(restatement LIKE '%{safe}%' OR keywords_text LIKE '%{safe}%')")
            })
            .collect();
        let where_clause = conditions.join(" OR ");

        let results = table
            .query()
            .only_if(where_clause)
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("keyword search failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| Self::batch_to_entries(b))
            .collect())
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
    ) -> Result<Vec<MemoryEntry>> {
        if filter.is_empty() {
            return Ok(Vec::new());
        }

        let table = self.get_table().await?;
        if table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?
            == 0
        {
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

        let where_clause = conditions.join(" AND ");

        let results = table
            .query()
            .only_if(where_clause)
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("structured search failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| Self::batch_to_entries(b))
            .collect())
    }

    /// Retrieve all entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn get_all(&self) -> Result<Vec<MemoryEntry>> {
        let table = self.get_table().await?;
        let results = table
            .query()
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("get all failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| Self::batch_to_entries(b))
            .collect())
    }

    /// Retrieve all entries together with their embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub async fn get_all_with_vectors(&self) -> Result<Vec<(MemoryEntry, Vec<f32>)>> {
        let table = self.get_table().await?;
        let results = table
            .query()
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("get all with vectors failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        let mut pairs = Vec::new();
        for batch in &batches {
            let entries = Self::batch_to_entries(batch);
            let vectors = batch_to_vectors(batch, self.dimension);
            for (entry, vec) in entries.into_iter().zip(vectors) {
                pairs.push((entry, vec));
            }
        }
        Ok(pairs)
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
        let table = self.get_table().await?;
        let ids_csv: String = entry_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        let predicate = format!("entry_id IN ({ids_csv})");
        let count = entry_ids.len();
        table
            .delete(&predicate)
            .await
            .map_err(|e| Error::VectorStore(format!("delete entries failed: {e}")))?;
        tracing::info!(count, "deleted entries from vector store");
        Ok(count)
    }

    /// Count the total number of entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn count(&self) -> Result<usize> {
        let table = self.get_table().await?;
        table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))
    }

    /// Clear all data and reinitialize.
    ///
    /// # Errors
    ///
    /// Returns an error if the table cannot be dropped or recreated.
    pub async fn clear(&self) -> Result<()> {
        self.db
            .drop_table(&self.table_name, &[])
            .await
            .map_err(|e| Error::VectorStore(format!("drop table failed: {e}")))?;
        self.ensure_table().await?;
        tracing::info!(table = %self.table_name, "cleared vector store");
        Ok(())
    }

    /// Consolidate memory: decay old entries, merge near-duplicates, prune low-importance.
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
    ) -> Result<ConsolidationStats> {
        let pairs = self.get_all_with_vectors().await?;
        if pairs.is_empty() {
            return Ok(ConsolidationStats::default());
        }

        let t0 = std::time::Instant::now();
        let (entries, vectors): (Vec<MemoryEntry>, Vec<Vec<f32>>) = pairs.into_iter().unzip();
        let n = entries.len();
        let mut importance: Vec<f64> = vec![1.0; n];
        let mut dead: Vec<bool> = vec![false; n];

        let now = chrono::Utc::now();
        let max_age_secs = f64::from(max_age_days) * 86400.0;
        let mut decayed = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            let Some(ts) = entry.timestamp else { continue };
            let age = (now - ts).num_seconds() as f64;
            if age > max_age_secs {
                importance[i] *= decay_factor;
                if importance[i] < min_importance {
                    dead[i] = true;
                }
                decayed += 1;
            }
        }

        let mut merged = 0usize;
        let mut ids_to_delete: Vec<String> = Vec::new();
        for i in 0..n {
            if dead[i] {
                continue;
            }
            for j in (i + 1)..n {
                if dead[j] {
                    continue;
                }
                if cosine_similarity(&vectors[i], &vectors[j]) >= merge_threshold {
                    let loser = if importance[i] >= importance[j] { j } else { i };
                    dead[loser] = true;
                    ids_to_delete.push(entries[loser].id.to_string());
                    merged += 1;
                }
            }
        }

        let mut pruned = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            if !dead[i] && importance[i] < min_importance {
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

fn split_delimited(s: &str) -> Vec<String> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split("||")
            .map(|p| p.trim().to_owned())
            .filter(|p| !p.is_empty())
            .collect()
    }
}

fn join_delimited(items: &[String]) -> String {
    items.join("||")
}

fn escape_like(s: &str) -> String {
    s.replace('\'', "''")
        .replace('%', "\\%")
        .replace('_', "\\_")
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
