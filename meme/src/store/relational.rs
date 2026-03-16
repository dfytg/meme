//! SQLite-backed relational store for cross-session data.

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use uuid::Uuid;

use crate::error::Result;
use crate::model::{
    ConsolidationRun, CrossObservation, EventKind, ObservationType, RedactionLevel, Session,
    SessionEvent, SessionStatus, SessionSummary,
};

/// `SQLite` store for cross-session metadata (sessions, events, observations, summaries).
pub struct SqliteStore {
    conn: Connection,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore").finish_non_exhaustive()
    }
}

impl SqliteStore {
    /// Open or create a `SQLite` database at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or migrations fail.
    pub fn open(path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Open an in-memory database (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if migrations fail.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL DEFAULT 'default',
                content_session_id TEXT NOT NULL,
                memory_session_id  TEXT NOT NULL UNIQUE,
                project         TEXT NOT NULL,
                user_prompt     TEXT,
                started_at      TEXT NOT NULL,
                ended_at        TEXT,
                status          TEXT NOT NULL DEFAULT 'active',
                metadata_json   TEXT
            );

            CREATE TABLE IF NOT EXISTS session_events (
                event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_session_id TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                kind            TEXT NOT NULL,
                title           TEXT,
                payload_json    TEXT,
                redaction_level TEXT NOT NULL DEFAULT 'none',
                FOREIGN KEY (memory_session_id) REFERENCES sessions(memory_session_id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                obs_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_session_id TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                obs_type        TEXT NOT NULL,
                title           TEXT NOT NULL,
                subtitle        TEXT,
                facts_json      TEXT,
                narrative       TEXT,
                concepts_json   TEXT,
                files_json      TEXT,
                vector_ref      TEXT,
                FOREIGN KEY (memory_session_id) REFERENCES sessions(memory_session_id)
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                summary_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_session_id TEXT NOT NULL UNIQUE,
                timestamp       TEXT NOT NULL,
                request         TEXT,
                investigated    TEXT,
                learned         TEXT,
                completed       TEXT,
                next_steps      TEXT,
                vector_ref      TEXT,
                FOREIGN KEY (memory_session_id) REFERENCES sessions(memory_session_id)
            );

            CREATE TABLE IF NOT EXISTS memory_links (
                link_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_entry_id TEXT NOT NULL,
                source_kind     TEXT NOT NULL,
                source_id       INTEGER NOT NULL,
                score           REAL NOT NULL,
                timestamp       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_runs (
                run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                policy_json     TEXT,
                stats_json      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
            CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(memory_session_id);
            CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(memory_session_id);
            ",
        )?;
        Ok(())
    }

    /// Insert a new session record.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_session(&self, session: &Session) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sessions (tenant_id, content_session_id, memory_session_id, project, user_prompt, started_at, status, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                session.tenant_id,
                session.content_session_id,
                session.memory_session_id.to_string(),
                session.project,
                session.user_prompt,
                session.started_at.to_rfc3339(),
                format!("{:?}", session.status).to_lowercase(),
                session.metadata_json,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update session status and optionally set `ended_at`.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_session_status(
        &self,
        memory_session_id: &Uuid,
        status: SessionStatus,
        ended_at: Option<DateTime<Utc>>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET status = ?1, ended_at = ?2 WHERE memory_session_id = ?3",
            params![
                format!("{status:?}").to_lowercase(),
                ended_at.map(|t| t.to_rfc3339()),
                memory_session_id.to_string(),
            ],
        )?;
        Ok(())
    }

    /// Get a session by its memory session ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_session(&self, memory_session_id: &Uuid) -> Result<Option<Session>> {
        self.conn
            .query_row(
                "SELECT id, tenant_id, content_session_id, memory_session_id, project, user_prompt, started_at, ended_at, status, metadata_json
                 FROM sessions WHERE memory_session_id = ?1",
                params![memory_session_id.to_string()],
                |row| {
                    Ok(Session {
                        row_id: Some(row.get(0)?),
                        tenant_id: row.get(1)?,
                        content_session_id: row.get(2)?,
                        memory_session_id: parse_uuid(&row.get::<_, String>(3)?),
                        project: row.get(4)?,
                        user_prompt: row.get(5)?,
                        started_at: parse_datetime(&row.get::<_, String>(6)?),
                        ended_at: row.get::<_, Option<String>>(7)?.map(|s| parse_datetime(&s)),
                        status: parse_session_status(&row.get::<_, String>(8)?),
                        metadata_json: row.get(9)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// List recent sessions for a project.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_sessions(&self, project: &str, limit: usize) -> Result<Vec<Session>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tenant_id, content_session_id, memory_session_id, project, user_prompt, started_at, ended_at, status, metadata_json
             FROM sessions WHERE project = ?1 ORDER BY started_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project, limit as i64], |row| {
            Ok(Session {
                row_id: Some(row.get(0)?),
                tenant_id: row.get(1)?,
                content_session_id: row.get(2)?,
                memory_session_id: parse_uuid(&row.get::<_, String>(3)?),
                project: row.get(4)?,
                user_prompt: row.get(5)?,
                started_at: parse_datetime(&row.get::<_, String>(6)?),
                ended_at: row.get::<_, Option<String>>(7)?.map(|s| parse_datetime(&s)),
                status: parse_session_status(&row.get::<_, String>(8)?),
                metadata_json: row.get(9)?,
            })
        })?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    /// Insert a session event.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_event(&self, event: &SessionEvent) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO session_events (memory_session_id, timestamp, kind, title, payload_json, redaction_level)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                event.memory_session_id.to_string(),
                event.timestamp.to_rfc3339(),
                format!("{:?}", event.kind).to_lowercase(),
                event.title,
                event.payload_json,
                format!("{:?}", event.redaction_level).to_lowercase(),
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get all events for a session.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_events(&self, memory_session_id: &Uuid) -> Result<Vec<SessionEvent>> {
        let mut stmt = self.conn.prepare(
            "SELECT event_id, memory_session_id, timestamp, kind, title, payload_json, redaction_level
             FROM session_events WHERE memory_session_id = ?1 ORDER BY timestamp ASC",
        )?;
        let rows = stmt.query_map(params![memory_session_id.to_string()], |row| {
            Ok(SessionEvent {
                event_id: Some(row.get(0)?),
                memory_session_id: parse_uuid(&row.get::<_, String>(1)?),
                timestamp: parse_datetime(&row.get::<_, String>(2)?),
                kind: parse_event_kind(&row.get::<_, String>(3)?),
                title: row.get(4)?,
                payload_json: row.get(5)?,
                redaction_level: parse_redaction_level(&row.get::<_, String>(6)?),
            })
        })?;

        let mut events = Vec::new();
        for row in rows {
            events.push(row?);
        }
        Ok(events)
    }

    /// Insert an observation.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_observation(&self, obs: &CrossObservation) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO observations (memory_session_id, timestamp, obs_type, title, subtitle, facts_json, narrative, concepts_json, files_json, vector_ref)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                obs.memory_session_id.to_string(),
                obs.timestamp.to_rfc3339(),
                format!("{:?}", obs.obs_type).to_lowercase(),
                obs.title,
                obs.subtitle,
                obs.facts_json,
                obs.narrative,
                obs.concepts_json,
                obs.files_json,
                obs.vector_ref,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get observations for a session.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_observations(&self, memory_session_id: &Uuid) -> Result<Vec<CrossObservation>> {
        let mut stmt = self.conn.prepare(
            "SELECT obs_id, memory_session_id, timestamp, obs_type, title, subtitle, facts_json, narrative, concepts_json, files_json, vector_ref
             FROM observations WHERE memory_session_id = ?1 ORDER BY timestamp ASC",
        )?;
        let rows = stmt.query_map(params![memory_session_id.to_string()], |row| {
            Ok(CrossObservation {
                obs_id: Some(row.get(0)?),
                memory_session_id: parse_uuid(&row.get::<_, String>(1)?),
                timestamp: parse_datetime(&row.get::<_, String>(2)?),
                obs_type: parse_observation_type(&row.get::<_, String>(3)?),
                title: row.get(4)?,
                subtitle: row.get(5)?,
                facts_json: row.get(6)?,
                narrative: row.get(7)?,
                concepts_json: row.get(8)?,
                files_json: row.get(9)?,
                vector_ref: row.get(10)?,
            })
        })?;

        let mut observations = Vec::new();
        for row in rows {
            observations.push(row?);
        }
        Ok(observations)
    }

    /// Get recent observations across all sessions for a project.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_recent_observations(
        &self,
        project: &str,
        limit: usize,
    ) -> Result<Vec<CrossObservation>> {
        let mut stmt = self.conn.prepare(
            "SELECT o.obs_id, o.memory_session_id, o.timestamp, o.obs_type, o.title, o.subtitle, o.facts_json, o.narrative, o.concepts_json, o.files_json, o.vector_ref
             FROM observations o
             JOIN sessions s ON o.memory_session_id = s.memory_session_id
             WHERE s.project = ?1
             ORDER BY o.timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project, limit as i64], |row| {
            Ok(CrossObservation {
                obs_id: Some(row.get(0)?),
                memory_session_id: parse_uuid(&row.get::<_, String>(1)?),
                timestamp: parse_datetime(&row.get::<_, String>(2)?),
                obs_type: parse_observation_type(&row.get::<_, String>(3)?),
                title: row.get(4)?,
                subtitle: row.get(5)?,
                facts_json: row.get(6)?,
                narrative: row.get(7)?,
                concepts_json: row.get(8)?,
                files_json: row.get(9)?,
                vector_ref: row.get(10)?,
            })
        })?;

        let mut observations = Vec::new();
        for row in rows {
            observations.push(row?);
        }
        Ok(observations)
    }

    /// Insert or replace a session summary.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn upsert_summary(&self, summary: &SessionSummary) -> Result<i64> {
        self.conn.execute(
            "INSERT OR REPLACE INTO session_summaries (memory_session_id, timestamp, request, investigated, learned, completed, next_steps, vector_ref)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                summary.memory_session_id.to_string(),
                summary.timestamp.to_rfc3339(),
                summary.request,
                summary.investigated,
                summary.learned,
                summary.completed,
                summary.next_steps,
                summary.vector_ref,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get recent summaries for a project.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_recent_summaries(&self, project: &str, limit: usize) -> Result<Vec<SessionSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.summary_id, s.memory_session_id, s.timestamp, s.request, s.investigated, s.learned, s.completed, s.next_steps, s.vector_ref
             FROM session_summaries s
             JOIN sessions sess ON s.memory_session_id = sess.memory_session_id
             WHERE sess.project = ?1
             ORDER BY s.timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project, limit as i64], |row| {
            Ok(SessionSummary {
                summary_id: Some(row.get(0)?),
                memory_session_id: parse_uuid(&row.get::<_, String>(1)?),
                timestamp: parse_datetime(&row.get::<_, String>(2)?),
                request: row.get(3)?,
                investigated: row.get(4)?,
                learned: row.get(5)?,
                completed: row.get(6)?,
                next_steps: row.get(7)?,
                vector_ref: row.get(8)?,
            })
        })?;

        let mut summaries = Vec::new();
        for row in rows {
            summaries.push(row?);
        }
        Ok(summaries)
    }

    /// Record a consolidation run.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_consolidation_run(&self, run: &ConsolidationRun) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO consolidation_runs (tenant_id, timestamp, policy_json, stats_json)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                run.tenant_id,
                run.timestamp.to_rfc3339(),
                run.policy_json,
                run.stats_json,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }
}

fn parse_uuid(s: &str) -> Uuid {
    Uuid::parse_str(s).unwrap_or_else(|_| Uuid::new_v4())
}

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s).map_or_else(|_| Utc::now(), |dt| dt.with_timezone(&Utc))
}

fn parse_session_status(s: &str) -> SessionStatus {
    match s {
        "completed" => SessionStatus::Completed,
        "failed" => SessionStatus::Failed,
        _ => SessionStatus::Active,
    }
}

fn parse_event_kind(s: &str) -> EventKind {
    match s {
        "message" => EventKind::Message,
        "tool_use" => EventKind::ToolUse,
        "file_change" => EventKind::FileChange,
        "note" => EventKind::Note,
        _ => EventKind::System,
    }
}

fn parse_redaction_level(s: &str) -> RedactionLevel {
    match s {
        "partial" => RedactionLevel::Partial,
        "full" => RedactionLevel::Full,
        _ => RedactionLevel::None,
    }
}

fn parse_observation_type(s: &str) -> ObservationType {
    match s {
        "decision" => ObservationType::Decision,
        "bugfix" => ObservationType::Bugfix,
        "feature" => ObservationType::Feature,
        "refactor" => ObservationType::Refactor,
        "discovery" => ObservationType::Discovery,
        _ => ObservationType::Change,
    }
}
