//! `meme count` — count stored memory entries.

use clap::Args;

use super::Context;

/// Count stored memory entries.
#[derive(Debug, Args)]
pub struct CountCmd;

impl CountCmd {
    /// Execute the count command.
    ///
    /// # Errors
    ///
    /// Returns an error if the count operation fails.
    pub async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let meme = ctx.build_meme().await?;
        let count = meme.count().await?;
        println!("{count}");
        Ok(())
    }
}
