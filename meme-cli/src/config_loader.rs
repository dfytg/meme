//! Configuration I/O — file loading, environment variable overrides, saving.
//!
//! This module owns all filesystem and environment interactions for
//! configuration.  The library crate (`meme`) only provides pure data
//! structures; the CLI is responsible for resolving paths, reading files,
//! and applying environment variable overrides.

use std::path::{Path, PathBuf};

use meme::config::Config;

/// Returns the default data directory (`~/.meme/`).
#[must_use]
pub fn default_data_dir() -> PathBuf {
    directories::BaseDirs::new()
        .map_or_else(|| PathBuf::from(".meme"), |d| d.home_dir().join(".meme"))
}

/// Returns the default config file path (`~/.meme/config.toml`).
#[must_use]
pub fn default_config_path() -> PathBuf {
    default_data_dir().join("config.toml")
}

/// Load configuration from a TOML file, with environment variable overrides.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn from_file(path: &Path) -> anyhow::Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let mut config: Config = toml::from_str(&content)?;
    apply_env_overrides(&mut config);
    Ok(config)
}

/// Load configuration from the default location (`~/.meme/config.toml`),
/// falling back to defaults if the file does not exist.
///
/// Always applies environment variable overrides on top.
#[must_use]
pub fn load_default() -> Config {
    let path = default_config_path();
    let mut config = if path.exists() {
        match from_file(&path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "config parse failed, using defaults");
                let mut c = Config::default();
                apply_env_overrides(&mut c);
                c
            }
        }
    } else {
        let mut c = Config::default();
        apply_env_overrides(&mut c);
        c
    };
    resolve_store_paths(&mut config);
    config
}

/// Save configuration to a TOML file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn save(config: &Config, path: &Path) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(config)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}

/// Apply environment variable overrides to a configuration.
pub fn apply_env_overrides(config: &mut Config) {
    if let Ok(v) = std::env::var("MEME_LLM_API_KEY") {
        config.llm.api_key = Some(v);
    }
    if let Ok(v) = std::env::var("MEME_LLM_BASE_URL") {
        config.llm.base_url = v;
    }
    if let Ok(v) = std::env::var("MEME_LLM_MODEL") {
        config.llm.model = v;
    }
    if let Ok(v) = std::env::var("MEME_EMBEDDING_PROVIDER") {
        config.embedding.provider = v.parse().unwrap_or(config.embedding.provider);
    }
}

/// Resolve relative store paths to absolute paths under `~/.meme/`.
fn resolve_store_paths(config: &mut Config) {
    let base = default_data_dir();
    if config.store.lancedb_path == Path::new(".meme/lancedb") {
        config.store.lancedb_path = base.join("lancedb");
    }
    if config.store.history_db_path == Path::new(".meme/history.db") {
        config.store.history_db_path = base.join("history.db");
    }
}
