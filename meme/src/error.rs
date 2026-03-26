//! Unified error types for the meme library.

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for the meme library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// LLM API call failed.
    #[error("llm error: {message}")]
    Llm {
        /// Human-readable error description.
        message: String,
        /// HTTP status code from the API, if available.
        status: Option<u16>,
        /// Whether this error is transient and the request can be retried.
        retryable: bool,
    },

    /// Failed to parse JSON from LLM response.
    #[error("json parse error: {0}")]
    JsonParse(String),

    /// Embedding computation failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Vector store operation failed.
    #[error("vector store error: {0}")]
    VectorStore(String),

    /// History store (`SQLite`) operation failed.
    #[error("history error: {0}")]
    History(String),

    /// Memory entry not found.
    #[error("not found: {id}")]
    NotFound {
        /// The ID that was not found.
        id: String,
    },

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Input validation error.
    #[error("validation error: {message}")]
    Validation {
        /// Description of the validation failure.
        message: String,
        /// The field that failed validation, if applicable.
        field: Option<String>,
    },

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

    /// Generic internal error.
    #[error("{0}")]
    Internal(String),
}

impl Error {
    /// Create an LLM error from a status code and message.
    pub fn llm(message: impl Into<String>) -> Self {
        Self::Llm {
            message: message.into(),
            status: None,
            retryable: false,
        }
    }

    /// Create an LLM error with HTTP status context.
    pub fn llm_with_status(status: u16, message: impl Into<String>) -> Self {
        let retryable = matches!(status, 429 | 500 | 502 | 503 | 504);
        Self::Llm {
            message: message.into(),
            status: Some(status),
            retryable,
        }
    }

    /// Create a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field: None,
        }
    }

    /// Create a vector store error from an Arrow error.
    pub fn arrow(e: impl std::fmt::Display) -> Self {
        Self::VectorStore(e.to_string())
    }

    /// Whether this error is transient and the operation can be retried.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Llm { retryable, .. } => *retryable,
            Self::Http(_) => true,
            _ => false,
        }
    }
}

impl From<lancedb::Error> for Error {
    fn from(e: lancedb::Error) -> Self {
        Self::VectorStore(e.to_string())
    }
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Self::History(e.to_string())
    }
}
