//! CLI subcommands.

pub(crate) mod add;
pub(crate) mod ask;
pub(crate) mod clear;
pub(crate) mod config;
pub(crate) mod consolidate;
pub(crate) mod count;
pub(crate) mod data;
pub(crate) mod delete;
pub(crate) mod get;
pub(crate) mod history;
pub(crate) mod init;
pub(crate) mod list;
pub(crate) mod search;
pub(crate) mod update;

use std::path::PathBuf;

use meme::Meme;
use uuid::Uuid;

use crate::config_loader;

/// Global options shared across all subcommands.
#[derive(Debug)]
pub(crate) struct Context {
    /// Namespace for memory isolation.
    pub(crate) namespace: Option<String>,
    /// Custom config file path.
    pub(crate) config_path: Option<PathBuf>,
}

impl Context {
    /// Build a [`Meme`] instance using the global options.
    ///
    /// Loads config from `--config` path or `~/.meme/config.toml`, applies
    /// environment variable overrides, then constructs the `Meme` instance.
    pub(crate) async fn build_meme(&self) -> anyhow::Result<Meme> {
        let config = match &self.config_path {
            Some(path) => config_loader::from_file(path)?,
            None => config_loader::load_default(),
        };
        let mut builder = Meme::builder().config(config);
        if let Some(ns) = &self.namespace {
            builder = builder.namespace(ns);
        }
        builder.build().await.map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Parse a UUID string, returning a user-friendly error on failure.
pub(crate) fn parse_uuid(s: &str) -> anyhow::Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| anyhow::anyhow!("invalid UUID '{s}': {e}"))
}

/// UTF-8 safe string truncation for display.
pub(crate) fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_owned()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}
