//! `meme export` / `meme import` — data import/export.

use clap::Args;

use super::Context;

/// Export memory entries to JSON.
#[derive(Debug, Args)]
pub(crate) struct ExportCmd {
    /// Output file path (stdout if not specified).
    #[arg(short, long)]
    pub output: Option<String>,
}

impl ExportCmd {
    /// Execute the export command.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let meme = ctx.build_meme().await?;
        let entries = meme.list().await?;
        let json = serde_json::to_string_pretty(&entries)?;

        if let Some(path) = &self.output {
            std::fs::write(path, &json)?;
            println!("Exported {} entries to {path}", entries.len());
        } else {
            println!("{json}");
        }
        Ok(())
    }
}

/// Import memory entries from a JSON file.
#[derive(Debug, Args)]
pub(crate) struct ImportCmd {
    /// Input file path.
    pub file: String,
}

impl ImportCmd {
    /// Execute the import command.
    ///
    /// # Errors
    ///
    /// Returns an error if the import fails.
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(&self.file)?;
        let entries: Vec<meme::Memory> = serde_json::from_str(&content)?;

        let count = entries.len();
        println!(
            "Importing {count} entries from {} (recomputing embeddings)...",
            self.file
        );

        let meme = ctx.build_meme().await?;
        meme.import(&entries).await?;

        println!("Imported {count} entries successfully.");
        Ok(())
    }
}
