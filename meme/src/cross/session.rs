//! Session manager — full session lifecycle orchestration.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::model::{Session, SessionStatus};
use crate::store::SqliteStore;

/// Manages the lifecycle of memory sessions (start → record → stop → end).
pub struct SessionManager<'a> {
    db: &'a SqliteStore,
}

impl std::fmt::Debug for SessionManager<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager").finish_non_exhaustive()
    }
}

impl<'a> SessionManager<'a> {
    /// Create a new session manager backed by the given `SQLite` store.
    pub const fn new(db: &'a SqliteStore) -> Self {
        Self { db }
    }

    /// Start a new session.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn start(
        &self,
        content_session_id: &str,
        project: &str,
        user_prompt: Option<&str>,
        tenant_id: &str,
    ) -> Result<Session> {
        let session = Session {
            row_id: None,
            tenant_id: tenant_id.to_owned(),
            content_session_id: content_session_id.to_owned(),
            memory_session_id: Uuid::new_v4(),
            project: project.to_owned(),
            user_prompt: user_prompt.map(String::from),
            started_at: Utc::now(),
            ended_at: None,
            status: SessionStatus::Active,
            metadata_json: None,
        };

        let row_id = self.db.insert_session(&session)?;
        tracing::info!(
            session_id = %session.memory_session_id,
            row_id,
            "session started"
        );

        Ok(Session {
            row_id: Some(row_id),
            ..session
        })
    }

    /// Mark a session as completed.
    ///
    /// # Errors
    ///
    /// Returns an error if the session is not found or already completed.
    pub fn stop(&self, memory_session_id: &Uuid) -> Result<()> {
        let session = self
            .db
            .get_session(memory_session_id)?
            .ok_or_else(|| Error::Session(format!("session {memory_session_id} not found")))?;

        if session.status != SessionStatus::Active {
            return Err(Error::Session(format!(
                "session {memory_session_id} is not active (status: {:?})",
                session.status
            )));
        }

        self.db.update_session_status(
            memory_session_id,
            SessionStatus::Completed,
            Some(Utc::now()),
        )?;

        tracing::info!(session_id = %memory_session_id, "session stopped");
        Ok(())
    }

    /// Mark a session as failed.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn fail(&self, memory_session_id: &Uuid) -> Result<()> {
        self.db.update_session_status(
            memory_session_id,
            SessionStatus::Failed,
            Some(Utc::now()),
        )?;
        tracing::info!(session_id = %memory_session_id, "session marked as failed");
        Ok(())
    }

    /// Get a session by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get(&self, memory_session_id: &Uuid) -> Result<Option<Session>> {
        self.db.get_session(memory_session_id)
    }

    /// List recent sessions for a project.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list(&self, project: &str, limit: usize) -> Result<Vec<Session>> {
        self.db.list_sessions(project, limit)
    }
}
