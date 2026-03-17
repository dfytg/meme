//! `meme delete` — delete a memory entry by UUID.

use clap::Args;

/// Delete a memory entry by its UUID.
#[derive(Debug, Args)]
pub struct DeleteCmd {
    /// Memory entry UUID.
    pub id: String,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl DeleteCmd {
    /// Execute the delete command.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;

        let uuid = uuid::Uuid::parse_str(&self.id)
            .map_err(|e| anyhow::anyhow!("invalid UUID '{id}': {e}", id = self.id))?;

        meme.delete(uuid)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("Deleted entry {uuid}");
        Ok(())
    }
}
