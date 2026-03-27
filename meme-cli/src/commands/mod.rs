//! CLI subcommands.

pub mod add;
pub mod ask;
pub mod clear;
pub mod config;
pub mod consolidate;
pub mod count;
pub mod data;
pub mod delete;
pub mod get;
pub mod history;
pub mod init;
pub mod list;
pub mod search;
pub mod update;

use std::path::PathBuf;

use meme::Meme;
use uuid::Uuid;

use crate::config_loader;

/// Global options shared across all subcommands.
#[derive(Debug)]
pub struct Context {
    /// Namespace for memory isolation.
    pub namespace: Option<String>,
    /// Custom config file path.
    pub config_path: Option<PathBuf>,
}

impl Context {
    /// Build a [`Meme`] instance using the global options.
    ///
    /// Loads config from `--config` path or `~/.meme/config.toml`, applies
    /// environment variable overrides, then constructs the `Meme` instance.
    pub async fn build_meme(&self) -> anyhow::Result<Meme> {
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
pub fn parse_uuid(s: &str) -> anyhow::Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| anyhow::anyhow!("invalid UUID '{s}': {e}"))
}

/// UTF-8 safe string truncation for display.
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_owned()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}
