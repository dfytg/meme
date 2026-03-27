//! `meme delete` — delete a memory entry by UUID.

use clap::Args;

use super::Context;

/// Delete a memory entry by its UUID.
#[derive(Debug, Args)]
pub struct DeleteCmd {
    /// Memory entry UUID.
    pub id: String,
}

impl DeleteCmd {
    /// Execute the delete command.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let uuid = super::parse_uuid(&self.id)?;
        let meme = ctx.build_meme().await?;
        meme.delete(uuid).await?;
        println!("Deleted entry {uuid}");
        Ok(())
    }
}
