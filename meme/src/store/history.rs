//! Memory history store — tracks all lifecycle events (add/update/delete).
//!
//! Uses `SQLite` with WAL mode for ACID-compliant, low-latency persistence.

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::Connection;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::model::{EventType, MemoryEvent};

/// Persistent store for memory lifecycle events backed by `SQLite`.
pub struct HistoryStore {
    conn: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for HistoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistoryStore").finish_non_exhaustive()
    }
}

impl HistoryStore {
    /// Open (or create) the `SQLite` history database at the given path.
    ///
    /// Enables WAL mode and creates the schema if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or migrated.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| Error::History(format!("open {}: {e}", path.display())))?;

        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| Error::History(format!("set WAL mode: {e}")))?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| Error::History(format!("set synchronous: {e}")))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| Error::History(format!("enable foreign keys: {e}")))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                memory_id   TEXT NOT NULL,
                event_type  TEXT NOT NULL CHECK(event_type IN ('add', 'update', 'delete')),
                old_content TEXT,
                new_content TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_memory_id  ON events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);",
        )
        .map_err(|e| Error::History(format!("migrate schema: {e}")))?;

        tracing::info!(path = %path.display(), "history store opened");

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Record a memory lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
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

        let conn = Arc::clone(&self.conn);
        let e = event.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = conn.lock().expect("history db lock poisoned");
            conn.execute(
                "INSERT INTO events (event_id, memory_id, event_type, old_content, new_content, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    e.id.to_string(),
                    e.memory_id.to_string(),
                    e.event_type.as_str(),
                    e.old_content,
                    e.new_content,
                    e.timestamp.to_rfc3339(),
                ],
            )
            .map_err(|err| Error::History(format!("insert event: {err}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::History(format!("spawn_blocking: {e}")))??;

        Ok(event)
    }

    /// Get all history events for a specific memory entry, ordered by time.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub async fn get_history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("history db lock poisoned");
            let mut stmt = conn
                .prepare(
                    "SELECT event_id, memory_id, event_type, old_content, new_content, created_at
                     FROM events
                     WHERE memory_id = ?1
                     ORDER BY created_at ASC",
                )
                .map_err(|e| Error::History(format!("prepare query: {e}")))?;

            let rows = stmt
                .query_map(rusqlite::params![memory_id.to_string()], |row| {
                    Ok(RawEvent {
                        event_id: row.get(0)?,
                        memory_id: row.get(1)?,
                        event_type: row.get(2)?,
                        old_content: row.get(3)?,
                        new_content: row.get(4)?,
                        created_at: row.get(5)?,
                    })
                })
                .map_err(|e| Error::History(format!("query events: {e}")))?;

            let mut events = Vec::new();
            for row in rows {
                let raw = row.map_err(|e| Error::History(format!("read row: {e}")))?;
                events.push(raw.into_event());
            }
            Ok(events)
        })
        .await
        .map_err(|e| Error::History(format!("spawn_blocking: {e}")))?
    }
}

impl EventType {
    /// String representation for `SQLite` storage.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "update" => Self::Update,
            "delete" => Self::Delete,
            _ => Self::Add,
        }
    }
}

/// Intermediate row type for `SQLite` → `MemoryEvent` conversion.
struct RawEvent {
    event_id: String,
    memory_id: String,
    event_type: String,
    old_content: Option<String>,
    new_content: Option<String>,
    created_at: String,
}

impl RawEvent {
    fn into_event(self) -> MemoryEvent {
        MemoryEvent {
            id: Uuid::parse_str(&self.event_id).unwrap_or_else(|_| Uuid::new_v4()),
            memory_id: Uuid::parse_str(&self.memory_id).unwrap_or_else(|_| Uuid::new_v4()),
            event_type: EventType::from_str(&self.event_type),
            old_content: self.old_content,
            new_content: self.new_content,
            timestamp: chrono::DateTime::parse_from_rfc3339(&self.created_at)
                .map_or_else(|_| Utc::now(), |dt| dt.with_timezone(&Utc)),
        }
    }
}
