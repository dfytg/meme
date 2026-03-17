//! `meme ask` — query the memory system.

use clap::Args;

/// Ask a question against stored memories.
#[derive(Debug, Args)]
pub struct AskCmd {
    /// The question to ask.
    pub question: String,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl AskCmd {
    /// Execute the ask command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;

        let answer = meme.ask(&self.question).await?;

        println!("{answer}");
        Ok(())
    }
}
