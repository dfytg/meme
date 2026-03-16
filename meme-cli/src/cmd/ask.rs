//! `meme ask` — query the memory system.

use clap::Args;

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
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = meme::MemeBuilder::new()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let answer = meme
            .ask(&self.question)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("{answer}");
        Ok(())
    }
}
