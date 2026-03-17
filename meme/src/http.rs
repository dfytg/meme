//! Shared HTTP client with production-ready defaults.

use std::time::Duration;

use crate::error::{Error, Result};

/// Default request timeout (60 seconds).
const DEFAULT_TIMEOUT: Duration = Duration::from_mins(1);

/// Default connection timeout (10 seconds).
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default max idle connections per host.
const DEFAULT_POOL_IDLE_PER_HOST: usize = 10;

/// Build a shared [`reqwest::Client`] with production-ready defaults.
///
/// - 60s request timeout
/// - 10s connection timeout
/// - Connection pooling (10 idle per host)
/// - Descriptive User-Agent header
///
/// # Errors
///
/// Returns an error if the HTTP client cannot be built.
pub fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(DEFAULT_TIMEOUT)
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .pool_max_idle_per_host(DEFAULT_POOL_IDLE_PER_HOST)
        .user_agent(concat!("meme/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| Error::Internal(format!("failed to build HTTP client: {e}")))
}
