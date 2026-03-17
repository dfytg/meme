//! `meme update` — update a memory entry's content.

use clap::Args;

/// Update a memory entry's content by UUID.
#[derive(Debug, Args)]
pub struct UpdateCmd {
    /// Memory entry UUID.
    pub id: String,

    /// New content for the memory entry.
    pub content: String,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl UpdateCmd {
    /// Execute the update command.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;

        let uuid = uuid::Uuid::parse_str(&self.id)
            .map_err(|e| anyhow::anyhow!("invalid UUID '{id}': {e}", id = self.id))?;

        meme.update(uuid, &self.content).await?;

        println!("Updated entry {uuid}");
        Ok(())
    }
}
