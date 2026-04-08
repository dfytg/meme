//! Memory history store — tracks all lifecycle events (add/update/delete).
//!
//! Uses `SQLite` with WAL mode for ACID-compliant, low-latency persistence.
//! All queries are scoped by `namespace` for multi-tenant isolation.

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::Connection;
use uuid::Uuid;

use crate::error::{MemeError, Result};
use crate::model::{Event, EventType};

/// Persistent store for memory lifecycle events backed by `SQLite`.
pub struct HistoryStore {
    /// Shared `SQLite` connection (behind a mutex for thread safety).
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
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                memory_id   TEXT NOT NULL,
                event_type  TEXT NOT NULL CHECK(event_type IN ('add', 'update', 'delete')),
                old_content TEXT,
                new_content TEXT,
                namespace   TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_memory_id  ON events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_events_scope      ON events(namespace);
            CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);",
        )?;
        tracing::info!(path = %path.display(), "history store opened");
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Record a memory lifecycle event within a scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    ///
    pub async fn record(
        &self,
        memory_id: Uuid,
        event_type: EventType,
        old_content: Option<&str>,
        new_content: Option<&str>,
        namespace: Option<&str>,
    ) -> Result<Event> {
        let event = Event {
            id: Uuid::new_v4(),
            memory_id,
            event_type,
            old_content: old_content.map(String::from),
            new_content: new_content.map(String::from),
            timestamp: Utc::now(),
        };

        let conn = Arc::clone(&self.conn);
        let e = event.clone();
        let ns = namespace.map(String::from);
        tokio::task::spawn_blocking(move || -> Result<()> {
            let guard = conn.lock().map_err(|err| MemeError::Internal(format!("mutex poisoned: {err}")))?;
            guard.execute(
                "INSERT INTO events (event_id, memory_id, event_type, old_content, new_content, namespace, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    e.id.to_string(),
                    e.memory_id.to_string(),
                    e.event_type.as_str(),
                    e.old_content,
                    e.new_content,
                    ns,
                    e.timestamp.to_rfc3339(),
                ],
            )?;
            drop(guard);
            Ok(())
        })
        .await
        .map_err(|e| MemeError::Internal(format!("spawn_blocking: {e}")))??;

        Ok(event)
    }

    /// Get all history events for a memory entry within a scope, ordered by time.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    ///
    pub async fn get_history(
        &self,
        memory_id: Uuid,
        namespace: Option<&str>,
    ) -> Result<Vec<Event>> {
        let conn = Arc::clone(&self.conn);
        let mid = memory_id.to_string();
        let ns = namespace.map(String::from);
        tokio::task::spawn_blocking(move || {
            let guard = conn
                .lock()
                .map_err(|err| MemeError::Internal(format!("mutex poisoned: {err}")))?;
            let raw = fetch_raw_events(&guard, &mid, ns.as_deref())?;
            drop(guard);
            raw.into_iter()
                .map(RawEvent::try_into_event)
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| MemeError::Internal(format!("spawn_blocking: {e}")))?
    }
}

/// Query raw event rows from `SQLite` for a given memory id.
fn fetch_raw_events(conn: &Connection, mid: &str, ns: Option<&str>) -> Result<Vec<RawEvent>> {
    let mut stmt = conn.prepare(
        "SELECT event_id, memory_id, event_type, old_content, new_content, created_at
         FROM events
         WHERE memory_id = ?1
           AND (namespace IS ?2)
         ORDER BY created_at ASC",
    )?;
    let rows = stmt.query_map(rusqlite::params![mid, ns], |row| {
        Ok(RawEvent {
            event_id: row.get(0)?,
            memory_id: row.get(1)?,
            event_type: row.get(2)?,
            old_content: row.get(3)?,
            new_content: row.get(4)?,
            created_at: row.get(5)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

/// Intermediate row type for `SQLite` → `Event` conversion.
struct RawEvent {
    /// UUID of the event itself.
    event_id: String,
    /// UUID of the associated memory.
    memory_id: String,
    /// Event type string ("add", "update", "delete").
    event_type: String,
    /// Previous content (for updates and deletes).
    old_content: Option<String>,
    /// New content (for adds and updates).
    new_content: Option<String>,
    /// ISO-8601 creation timestamp.
    created_at: String,
}

impl RawEvent {
    /// Convert this raw row into a typed [`Event`].
    fn try_into_event(self) -> Result<Event> {
        Ok(Event {
            id: Uuid::parse_str(&self.event_id).map_err(|e| {
                MemeError::History(format!("corrupt event_id '{}': {e}", self.event_id))
            })?,
            memory_id: Uuid::parse_str(&self.memory_id).map_err(|e| {
                MemeError::History(format!("corrupt memory_id '{}': {e}", self.memory_id))
            })?,
            event_type: self.event_type.parse().map_err(|e| {
                MemeError::History(format!("corrupt event_type '{}': {e}", self.event_type))
            })?,
            old_content: self.old_content,
            new_content: self.new_content,
            timestamp: chrono::DateTime::parse_from_rfc3339(&self.created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| {
                    MemeError::History(format!("corrupt created_at '{}': {e}", self.created_at))
                })?,
        })
    }
}
