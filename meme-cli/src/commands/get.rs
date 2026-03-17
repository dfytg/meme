//! `meme get` — retrieve a single memory entry by UUID.

use clap::Args;

/// Get a memory entry by its UUID.
#[derive(Debug, Args)]
pub struct GetCmd {
    /// Memory entry UUID (full or prefix).
    pub id: String,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl GetCmd {
    /// Execute the get command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;

        let uuid = uuid::Uuid::parse_str(&self.id)
            .map_err(|e| anyhow::anyhow!("invalid UUID '{id}': {e}", id = self.id))?;

        let entry = meme
            .get(uuid)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .ok_or_else(|| anyhow::anyhow!("entry not found: {uuid}"))?;

        println!("{}", serde_json::to_string_pretty(&entry)?);
        Ok(())
    }
}
