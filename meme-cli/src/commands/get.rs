//! `meme get` — retrieve a single memory entry by UUID.

use clap::Args;

use super::Context;

/// Get a memory entry by its UUID.
#[derive(Debug, Args)]
pub struct GetCmd {
    /// Memory entry UUID.
    pub id: String,
}

impl GetCmd {
    /// Execute the get command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let uuid = super::parse_uuid(&self.id)?;
        let meme = ctx.build_meme().await?;
        let entry = meme
            .get(uuid)
            .await?
            .ok_or_else(|| anyhow::anyhow!("entry not found: {uuid}"))?;
        println!("{}", serde_json::to_string_pretty(&entry)?);
        Ok(())
    }
}
