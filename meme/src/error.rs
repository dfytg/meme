//! Unified error types for the meme library.

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for the meme library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// LLM API call failed.
    #[error("llm error: {0}")]
    Llm(String),

    /// Failed to parse JSON from LLM response.
    #[error("json parse error: {0}")]
    JsonParse(String),

    /// Embedding computation failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Vector store operation failed.
    #[error("vector store error: {0}")]
    VectorStore(String),

    /// `SQLite` operation failed.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP request error.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Serialization/deserialization error.
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    /// TOML deserialization error.
    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Session not found or invalid state.
    #[error("session error: {0}")]
    Session(String),

    /// Generic internal error.
    #[error("{0}")]
    Internal(String),
}
