//! Namespace-based memory isolation.

/// Opaque namespace for memory isolation.
///
/// When set, all queries are automatically filtered to only return entries
/// belonging to the specified namespace.  The library treats the value as an
/// opaque string — callers decide the semantics (user ID, session ID,
/// composite key, etc.).
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// Namespace identifier.  `None` means "global / unscoped".
    pub namespace: Option<String>,
}

impl Scope {
    /// Create a scoped namespace.
    #[must_use]
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
        }
    }

    /// Returns `true` if no namespace is set.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.namespace.is_none()
    }
}
