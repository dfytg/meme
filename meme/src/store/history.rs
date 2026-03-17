//! Memory history store — tracks all lifecycle events (add/update/delete).
//!
//! Uses a separate `LanceDB` table for zero-dependency persistence.

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::model::{EventType, MemoryEvent};

/// Persistent store for memory lifecycle events.
pub struct HistoryStore {
    db: lancedb::Connection,
    table_name: String,
}

impl std::fmt::Debug for HistoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistoryStore")
            .field("table_name", &self.table_name)
            .finish_non_exhaustive()
    }
}

impl HistoryStore {
    /// Open or create the history table.
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails.
    pub async fn open(db: lancedb::Connection, table_name: &str) -> Result<Self> {
        let store = Self {
            db,
            table_name: table_name.to_owned(),
        };
        store.ensure_table().await?;
        Ok(store)
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("memory_id", DataType::Utf8, false),
            Field::new("event_type", DataType::Utf8, false),
            Field::new("old_content", DataType::Utf8, true),
            Field::new("new_content", DataType::Utf8, true),
            Field::new("timestamp", DataType::Utf8, false),
        ]))
    }

    async fn ensure_table(&self) -> Result<()> {
        let tables = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("list tables failed: {e}")))?;

        if !tables.contains(&self.table_name) {
            self.db
                .create_empty_table(&self.table_name, Self::schema())
                .execute()
                .await
                .map_err(|e| Error::VectorStore(format!("create history table failed: {e}")))?;
            tracing::info!(table = %self.table_name, "created history table");
        }
        Ok(())
    }

    async fn get_table(&self) -> Result<lancedb::Table> {
        self.db
            .open_table(&self.table_name)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("open history table failed: {e}")))
    }

    /// Record a memory lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub async fn record(
        &self,
        memory_id: Uuid,
        event_type: EventType,
        old_content: Option<&str>,
        new_content: Option<&str>,
    ) -> Result<MemoryEvent> {
        let event = MemoryEvent {
            id: Uuid::new_v4(),
            memory_id,
            event_type,
            old_content: old_content.map(String::from),
            new_content: new_content.map(String::from),
            timestamp: Utc::now(),
        };

        let schema = Self::schema();
        let event_type_str = match event_type {
            EventType::Add => "add",
            EventType::Update => "update",
            EventType::Delete => "delete",
        };

        let col_event_id: Arc<dyn Array> = Arc::new(StringArray::from(vec![event.id.to_string()]));
        let col_memory_id: Arc<dyn Array> =
            Arc::new(StringArray::from(vec![event.memory_id.to_string()]));
        let col_event_type: Arc<dyn Array> =
            Arc::new(StringArray::from(vec![event_type_str.to_owned()]));
        let col_old: Arc<dyn Array> = Arc::new(StringArray::from(vec![
            old_content.unwrap_or_default().to_owned(),
        ]));
        let col_new: Arc<dyn Array> = Arc::new(StringArray::from(vec![
            new_content.unwrap_or_default().to_owned(),
        ]));
        let col_ts: Arc<dyn Array> =
            Arc::new(StringArray::from(vec![event.timestamp.to_rfc3339()]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                col_event_id,
                col_memory_id,
                col_event_type,
                col_old,
                col_new,
                col_ts,
            ],
        )
        .map_err(|e| Error::VectorStore(format!("history batch error: {e}")))?;

        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let table = self.get_table().await?;
        table
            .add(reader)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("record history failed: {e}")))?;

        Ok(event)
    }

    /// Get all history events for a specific memory entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        let table = self.get_table().await?;
        let filter = format!("memory_id = '{memory_id}'");
        let results = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| Error::VectorStore(format!("query history failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| Error::VectorStore(format!("collect history failed: {e}")))?;

        let mut events = Vec::new();
        for batch in &batches {
            events.extend(Self::batch_to_events(batch));
        }
        events.sort_by_key(|e| e.timestamp);
        Ok(events)
    }

    fn batch_to_events(batch: &RecordBatch) -> Vec<MemoryEvent> {
        let n = batch.num_rows();
        let get_str = |name: &str| -> Option<&StringArray> {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        };

        let event_id_col = get_str("event_id");
        let memory_id_col = get_str("memory_id");
        let event_type_col = get_str("event_type");
        let old_content_col = get_str("old_content");
        let new_content_col = get_str("new_content");
        let ts_col = get_str("timestamp");

        let mut events = Vec::with_capacity(n);
        for i in 0..n {
            let event_id = event_id_col
                .map(|c| c.value(i))
                .and_then(|s| Uuid::parse_str(s).ok())
                .unwrap_or_else(Uuid::new_v4);
            let memory_id = memory_id_col
                .map(|c| c.value(i))
                .and_then(|s| Uuid::parse_str(s).ok())
                .unwrap_or_else(Uuid::new_v4);
            let event_type = match event_type_col.map(|c| c.value(i)) {
                Some("update") => EventType::Update,
                Some("delete") => EventType::Delete,
                _ => EventType::Add,
            };
            let old_content = old_content_col
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i))
                .filter(|s| !s.is_empty())
                .map(String::from);
            let new_content = new_content_col
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i))
                .filter(|s| !s.is_empty())
                .map(String::from);
            let timestamp = ts_col
                .map(|c| c.value(i))
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map_or_else(Utc::now, |dt| dt.with_timezone(&Utc));

            events.push(MemoryEvent {
                id: event_id,
                memory_id,
                event_type,
                old_content,
                new_content,
                timestamp,
            });
        }
        events
    }
}
