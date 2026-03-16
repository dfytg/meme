//! Vector store — multi-view indexing with LanceDB + Tantivy FTS.

use std::sync::Arc;

use crate::embedding::EmbeddingProvider;
use crate::error::{Error, Result};
use crate::model::{MemoryEntry, MetadataFilter};

/// Trait for vector-based storage and retrieval of memory entries.
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Add entries with their pre-computed embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> Result<()>;

    /// Semantic search by vector similarity.
    ///
    /// # Errors
    ///
    /// Returns an error if the search operation fails.
    async fn semantic_search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<MemoryEntry>>;

    /// Keyword search using full-text search (BM25).
    ///
    /// # Errors
    ///
    /// Returns an error if the search operation fails.
    async fn keyword_search(&self, keywords: &[String], top_k: usize) -> Result<Vec<MemoryEntry>>;

    /// Structured search by metadata filtering.
    ///
    /// # Errors
    ///
    /// Returns an error if the search operation fails.
    async fn structured_search(
        &self,
        filter: &MetadataFilter,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>>;

    /// Retrieve all entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    async fn get_all(&self) -> Result<Vec<MemoryEntry>>;

    /// Count the total number of entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    async fn count(&self) -> Result<usize>;

    /// Clear all data and reinitialize.
    ///
    /// # Errors
    ///
    /// Returns an error if the clear operation fails.
    async fn clear(&self) -> Result<()>;
}

/// LanceDB-backed vector store with Tantivy FTS support.
pub struct LanceDbStore {
    db: lancedb::Connection,
    table_name: String,
    embedding: Arc<dyn EmbeddingProvider>,
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
            .map_err(|e| Error::VectorStore(format!("failed to connect to LanceDB: {e}")))?;

        let dimension = embedding.dimension();
        let store = Self {
            db,
            table_name: table_name.to_owned(),
            embedding,
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
            .map_err(|e| Error::VectorStore(format!("failed to list tables: {e}")))?;

        if !tables.contains(&self.table_name) {
            let schema = self.build_schema();
            self.db
                .create_empty_table(&self.table_name, schema)
                .execute()
                .await
                .map_err(|e| Error::VectorStore(format!("failed to create table: {e}")))?;
            tracing::info!(table = %self.table_name, "created new LanceDB table");
        } else {
            tracing::info!(table = %self.table_name, "opened existing LanceDB table");
        }
        Ok(())
    }

    fn build_schema(&self) -> arrow_schema::SchemaRef {
        use arrow_schema::{DataType, Field, Schema};

        Arc::new(Schema::new(vec![
            Field::new("entry_id", DataType::Utf8, false),
            Field::new("restatement", DataType::Utf8, false),
            Field::new(
                "keywords",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("timestamp", DataType::Utf8, true),
            Field::new("location", DataType::Utf8, true),
            Field::new(
                "persons",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new(
                "entities",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
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
            .map_err(|e| Error::VectorStore(format!("failed to open table: {e}")))
    }

    fn rows_to_entries(&self, batch: &arrow_array::RecordBatch) -> Vec<MemoryEntry> {
        use arrow_array::cast::AsArray;

        let n = batch.num_rows();
        let mut entries = Vec::with_capacity(n);

        let id_col = batch.column_by_name("entry_id");
        let restatement_col = batch.column_by_name("restatement");
        let keywords_col = batch.column_by_name("keywords");
        let timestamp_col = batch.column_by_name("timestamp");
        let location_col = batch.column_by_name("location");
        let persons_col = batch.column_by_name("persons");
        let entities_col = batch.column_by_name("entities");
        let topic_col = batch.column_by_name("topic");

        for i in 0..n {
            let id_str = id_col
                .and_then(|c| c.as_string_opt::<i32>())
                .and_then(|a| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(a.value(i).to_owned())
                    }
                })
                .unwrap_or_default();

            let restatement = restatement_col
                .and_then(|c| c.as_string_opt::<i32>())
                .and_then(|a| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(a.value(i).to_owned())
                    }
                })
                .unwrap_or_default();

            let keywords = extract_string_list(keywords_col, i);
            let persons = extract_string_list(persons_col, i);
            let entities = extract_string_list(entities_col, i);

            let timestamp_str = extract_optional_string(timestamp_col, i);
            let timestamp = timestamp_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            });

            let location = extract_optional_string(location_col, i);
            let topic = extract_optional_string(topic_col, i);

            let id = uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4());

            entries.push(MemoryEntry {
                id,
                restatement,
                keywords,
                timestamp,
                location,
                persons,
                entities,
                topic,
            });
        }

        entries
    }
}

fn extract_optional_string(col: Option<&arrow_array::ArrayRef>, idx: usize) -> Option<String> {
    use arrow_array::cast::AsArray;
    col.and_then(|c| c.as_string_opt::<i32>()).and_then(|a| {
        if a.is_null(idx) {
            None
        } else {
            let s = a.value(idx);
            if s.is_empty() {
                None
            } else {
                Some(s.to_owned())
            }
        }
    })
}

fn extract_string_list(col: Option<&arrow_array::ArrayRef>, idx: usize) -> Vec<String> {
    use arrow_array::cast::AsArray;
    col.and_then(|c| c.as_list_opt::<i32>())
        .map(|list_arr| {
            if list_arr.is_null(idx) {
                return Vec::new();
            }
            let values = list_arr.value(idx);
            let str_arr = values.as_string_opt::<i32>();
            match str_arr {
                Some(sa) => (0..sa.len())
                    .filter(|&j| !sa.is_null(j))
                    .map(|j| sa.value(j).to_owned())
                    .collect(),
                None => Vec::new(),
            }
        })
        .unwrap_or_default()
}

#[async_trait::async_trait]
impl VectorStore for LanceDbStore {
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        use std::sync::Arc;

        use arrow_array::{
            Array, ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray,
        };
        use arrow_schema::{DataType, Field};

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

        let keywords = build_list_array(entries.iter().map(|e| &e.keywords));
        let persons = build_list_array(entries.iter().map(|e| &e.persons));
        let entities_arr = build_list_array(entries.iter().map(|e| &e.entities));

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
        let topics: ArrayRef = Arc::new(StringArray::from(
            entries
                .iter()
                .map(|e| e.topic.clone().unwrap_or_default())
                .collect::<Vec<_>>(),
        ));

        // Build fixed-size list for vectors.
        let dim = self.dimension;
        let flat_values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let values_array = Float32Array::from(flat_values);
        let fsl_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(fsl_field, dim as i32, Arc::new(values_array), None)
                .map_err(|e| Error::VectorStore(format!("vector array error: {e}")))?,
        );

        let batch = RecordBatch::try_new(
            schema,
            vec![
                entry_ids,
                restatements,
                keywords,
                timestamps,
                locations,
                persons,
                entities_arr,
                topics,
                vector_array,
            ],
        )
        .map_err(|e| Error::VectorStore(format!("record batch error: {e}")))?;

        let table = self.get_table().await?;
        table
            .add(Box::new(arrow_array::RecordBatchIterator::new(
                vec![Ok(batch)],
                self.build_schema(),
            )))
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("add entries failed: {e}")))?;

        tracing::info!(count = n, "added memory entries to vector store");
        Ok(())
    }

    async fn semantic_search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<MemoryEntry>> {
        let table = self.get_table().await?;
        let count = table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let results = table
            .vector_search(query_vec)
            .map_err(|e| Error::VectorStore(format!("vector search init failed: {e}")))?
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("vector search failed: {e}")))?;

        use futures::TryStreamExt;
        let batches: Vec<arrow_array::RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect results failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.rows_to_entries(b))
            .collect())
    }

    async fn keyword_search(&self, keywords: &[String], top_k: usize) -> Result<Vec<MemoryEntry>> {
        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let table = self.get_table().await?;
        let count = table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let query = keywords.join(" ");
        let results = table
            .search(lancedb::query::QueryBase::FullTextSearch(
                lancedb::query::FullTextSearchQuery::new(query),
            ))
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("keyword search failed: {e}")))?;

        use futures::TryStreamExt;
        let batches: Vec<arrow_array::RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect results failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.rows_to_entries(b))
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
        let count = table
            .count_rows(None)
            .await
            .map_err(|e| Error::VectorStore(format!("count failed: {e}")))?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut conditions = Vec::new();

        if let Some(persons) = &filter.persons {
            let values = persons
                .iter()
                .map(|p| format!("'{}'", p.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(", ");
            conditions.push(format!("array_has_any(persons, make_array({values}))"));
        }

        if let Some(location) = &filter.location {
            let safe = location.replace('\'', "''");
            conditions.push(format!("location LIKE '%{safe}%'"));
        }

        if let Some(entities) = &filter.entities {
            let values = entities
                .iter()
                .map(|e| format!("'{}'", e.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(", ");
            conditions.push(format!("array_has_any(entities, make_array({values}))"));
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
        let batches: Vec<arrow_array::RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect results failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.rows_to_entries(b))
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
        let batches: Vec<arrow_array::RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect results failed: {e}")))?;

        Ok(batches
            .iter()
            .flat_map(|b| self.rows_to_entries(b))
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
        let _ = self
            .db
            .drop_table(&self.table_name)
            .await
            .map_err(|e| Error::VectorStore(format!("drop table failed: {e}")))?;
        self.ensure_table().await?;
        tracing::info!(table = %self.table_name, "cleared vector store");
        Ok(())
    }
}

fn build_list_array<'a>(items: impl Iterator<Item = &'a Vec<String>>) -> arrow_array::ArrayRef {
    use arrow_array::{Array, ListArray, StringArray};
    use arrow_schema::{DataType, Field};

    let items: Vec<&Vec<String>> = items.collect();
    let mut offsets = vec![0i32];
    let mut all_values = Vec::new();
    for list in &items {
        for s in *list {
            all_values.push(s.as_str());
        }
        offsets.push(all_values.len() as i32);
    }

    let values = Arc::new(StringArray::from(all_values));
    let offsets_buf = arrow_array::OffsetSizeTrait::from_usize;
    let offset_buffer = arrow_array::builder::BufferBuilder::<i32>::new(offsets.len());

    // Use ListArray::from_iter_primitive alternative: manual construction
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    Arc::new(
        ListArray::try_new(
            field,
            arrow_array::OffsetBuffer::new(offsets.into()),
            values,
            None,
        )
        .expect("valid list array"),
    )
}
