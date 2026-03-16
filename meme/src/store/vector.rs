//! Vector store — multi-view indexing with LanceDB.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::embedding::EmbeddingProvider;
use crate::error::{Error, Result};
use crate::model::{MemoryEntry, MetadataFilter};

/// Trait for vector-based storage and retrieval of memory entries.
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Add entries with their pre-computed embedding vectors.
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> Result<()>;

    /// Semantic search by vector similarity.
    async fn semantic_search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<MemoryEntry>>;

    /// Keyword search (scans restatement text for keyword matches).
    async fn keyword_search(&self, keywords: &[String], top_k: usize) -> Result<Vec<MemoryEntry>>;

    /// Structured search by metadata filtering.
    async fn structured_search(
        &self,
        filter: &MetadataFilter,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>>;

    /// Retrieve all entries.
    async fn get_all(&self) -> Result<Vec<MemoryEntry>>;

    /// Count the total number of entries.
    async fn count(&self) -> Result<usize>;

    /// Clear all data and reinitialize.
    async fn clear(&self) -> Result<()>;
}

/// LanceDB-backed vector store.
pub struct LanceDbStore {
    db: lancedb::Connection,
    table_name: String,
    dimension: usize,
}

impl std::fmt::Debug for LanceDbStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceDbStore")
            .field("table_name", &self.table_name)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl LanceDbStore {
    /// Open or create a LanceDB store.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn open(
        db_path: &str,
        table_name: &str,
        embedding: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;
        let db = lancedb::connect(db_path)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("failed to connect: {e}")))?;

        let dimension = embedding.dimension();
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

        if !tables.contains(&self.table_name) {
            let schema = self.build_schema();
            self.db
                .create_empty_table(&self.table_name, schema)
                .execute()
                .await
                .map_err(|e| Error::VectorStore(format!("create table failed: {e}")))?;
            tracing::info!(table = %self.table_name, "created new LanceDB table");
        } else {
            tracing::info!(table = %self.table_name, "opened existing LanceDB table");
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
                    self.dimension as i32,
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

    fn batch_to_entries(&self, batch: &RecordBatch) -> Vec<MemoryEntry> {
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

#[async_trait::async_trait]
impl VectorStore for LanceDbStore {
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> Result<()> {
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
            FixedSizeListArray::try_new(fsl_field, dim as i32, Arc::new(values), None)
                .map_err(|e| Error::VectorStore(format!("vector array error: {e}")))?,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
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

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let table = self.get_table().await?;
        let was_empty = table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?
            == 0;

        table
            .add(Box::new(reader))
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("add entries failed: {e}")))?;

        // Create FTS index after first data insertion.
        if was_empty {
            if let Err(e) = table
                .create_index(
                    &["restatement"],
                    lancedb::index::Index::FTS(Default::default()),
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

    async fn semantic_search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<MemoryEntry>> {
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

        use futures::TryStreamExt;
        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.batch_to_entries(b))
            .collect())
    }

    async fn keyword_search(&self, keywords: &[String], top_k: usize) -> Result<Vec<MemoryEntry>> {
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

        let conditions: Vec<String> = keywords
            .iter()
            .map(|kw| {
                let safe = kw.replace('\'', "''");
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

        use futures::TryStreamExt;
        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.batch_to_entries(b))
            .collect())
    }

    async fn structured_search(
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
                .map(|p| format!("persons_text LIKE '%{}%'", p.replace('\'', "''")))
                .collect();
            if !conds.is_empty() {
                conditions.push(format!("({})", conds.join(" OR ")));
            }
        }

        if let Some(location) = &filter.location {
            conditions.push(format!(
                "location LIKE '%{}%'",
                location.replace('\'', "''")
            ));
        }

        if let Some(entities) = &filter.entities {
            let conds: Vec<String> = entities
                .iter()
                .map(|e| format!("entities_text LIKE '%{}%'", e.replace('\'', "''")))
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

        use futures::TryStreamExt;
        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.batch_to_entries(b))
            .collect())
    }

    async fn get_all(&self) -> Result<Vec<MemoryEntry>> {
        let table = self.get_table().await?;
        let results = table
            .query()
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("get all failed: {e}")))?;

        use futures::TryStreamExt;
        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.batch_to_entries(b))
            .collect())
    }

    async fn count(&self) -> Result<usize> {
        let table = self.get_table().await?;
        table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))
    }

    async fn clear(&self) -> Result<()> {
        self.db
            .drop_table(&self.table_name, &[])
            .await
            .map_err(|e| Error::VectorStore(format!("drop table failed: {e}")))?;
        self.ensure_table().await?;
        tracing::info!(table = %self.table_name, "cleared vector store");
        Ok(())
    }
}
