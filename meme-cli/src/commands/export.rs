//! `meme export` / `meme import` — data import/export.

use clap::Args;

/// Export memory entries to JSON.
#[derive(Debug, Args)]
pub struct ExportCmd {
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
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = meme::MemeBuilder::new()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let entries = meme
            .get_all_memories()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

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
pub struct ImportCmd {
    /// Input file path.
    pub file: String,
}

impl ImportCmd {
    /// Execute the import command.
    ///
    /// # Errors
    ///
    /// Returns an error if the import fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(&self.file)?;
        let entries: Vec<meme::model::MemoryEntry> = serde_json::from_str(&content)?;

        let count = entries.len();
        println!("Parsed {count} entries from {}", self.file);
        println!("Note: Direct entry import requires embedding recomputation.");
        println!("Use `meme add --file` for JSONL dialogue import instead.");

        Ok(())
    }
}
