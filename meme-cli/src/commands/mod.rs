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

/// Build a [`Meme`] instance with optional scope parameters.
pub async fn build_meme(user_id: Option<&str>, session_id: Option<&str>) -> anyhow::Result<Meme> {
    let mut builder = MemeBuilder::new();
    if let Some(uid) = user_id {
        builder = builder.user_id(uid);
    }
    if let Some(sid) = session_id {
        builder = builder.session_id(sid);
    }
    builder.build().await.map_err(|e| anyhow::anyhow!("{e}"))
}
