//! `meme clear` — clear all stored memories.

use clap::Args;

use super::Context;

/// Clear all stored memories for the current scope.
#[derive(Debug, Args)]
pub(crate) struct ClearCmd {
    /// Skip confirmation prompt.
    #[arg(long)]
    pub force: bool,
}

impl ClearCmd {
    /// Execute the clear command.
    ///
    /// # Errors
    ///
    /// Returns an error if the clear operation fails.
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        if !self.force {
            eprint!("This will delete ALL memories. Are you sure? [y/N] ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Aborted.");
                return Ok(());
            }
        }

        let meme = ctx.build_meme().await?;
        let count = meme.count().await?;
        meme.clear().await?;
        println!("Cleared {count} memories.");
        Ok(())
    }
}
