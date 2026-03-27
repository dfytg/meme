//! `meme ask` — query the memory system.

use clap::Args;

use super::Context;

/// Ask a question against stored memories.
#[derive(Debug, Args)]
pub struct AskCmd {
    /// The question to ask.
    pub question: String,
}

impl AskCmd {
    /// Execute the ask command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let meme = ctx.build_meme().await?;
        let answer = meme.ask(&self.question).await?;
        println!("{answer}");
        Ok(())
    }
}
