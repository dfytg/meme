//! Tenant scope for multi-user / multi-session isolation.

/// Tenant scope for multi-user / multi-session isolation.
///
/// When set, all queries are automatically filtered to only return entries
/// belonging to the specified user and/or session.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// Filter by user identifier.
    pub user_id: Option<String>,
    /// Filter by session identifier.
    pub session_id: Option<String>,
}
