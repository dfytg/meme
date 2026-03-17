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
        let mut builder = meme::MemeBuilder::new();
        if let Some(uid) = &self.user_id {
            builder = builder.user_id(uid);
        }
        if let Some(sid) = &self.session_id {
            builder = builder.session_id(sid);
        }
        let meme = builder.build().await.map_err(|e| anyhow::anyhow!("{e}"))?;

        let answer = meme
            .ask(&self.question)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("{answer}");
        Ok(())
    }
}
