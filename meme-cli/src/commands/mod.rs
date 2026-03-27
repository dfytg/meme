//! CLI subcommands.

pub mod add;
pub mod ask;
pub mod delete;
pub mod export;
pub mod get;
pub mod history;
pub mod init;
pub mod list;
pub mod search;
pub mod update;

use meme::{Meme, MemeBuilder};

use crate::config_loader;

/// UTF-8 safe string truncation for display.
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_owned()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

/// Build a [`Meme`] instance with optional scope parameters.
///
/// Loads configuration from `~/.meme/config.toml` (if present) with
/// environment variable overrides, then constructs the `Meme` instance.
pub async fn build_meme(user_id: Option<&str>, session_id: Option<&str>) -> anyhow::Result<Meme> {
    let config = config_loader::load_default();
    let mut builder = MemeBuilder::new().config(config);
    if let Some(uid) = user_id {
        builder = builder.user_id(uid);
    }
    if let Some(sid) = session_id {
        builder = builder.session_id(sid);
    }
    let meme = builder.build().await.map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(meme)
}
