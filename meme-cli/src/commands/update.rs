//! `meme update` — update a memory entry's content.

use clap::Args;

use super::Context;

/// Update a memory entry's content by UUID.
#[derive(Debug, Args)]
pub(crate) struct UpdateCmd {
    /// Memory entry UUID.
    pub id: String,

    /// New content for the memory entry.
    pub content: String,
}

impl UpdateCmd {
    /// Execute the update command.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let uuid = super::parse_uuid(&self.id)?;
        let meme = ctx.build_meme().await?;
        meme.update(uuid, &self.content).await?;
        println!("Updated entry {uuid}");
        Ok(())
    }
}
