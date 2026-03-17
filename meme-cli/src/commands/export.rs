//! `meme export` / `meme import` — data import/export.

use clap::Args;

/// Export memory entries to JSON.
#[derive(Debug, Args)]
pub struct ExportCmd {
    /// Output file path (stdout if not specified).
    #[arg(short, long)]
    pub output: Option<String>,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl ExportCmd {
    /// Execute the export command.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let mut builder = meme::MemeBuilder::new();
        if let Some(uid) = &self.user_id {
            builder = builder.user_id(uid);
        }
        if let Some(sid) = &self.session_id {
            builder = builder.session_id(sid);
        }
        let meme = builder.build().await.map_err(|e| anyhow::anyhow!("{e}"))?;

        let entries = meme.get_all().await.map_err(|e| anyhow::anyhow!("{e}"))?;

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

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl ImportCmd {
    /// Execute the import command.
    ///
    /// # Errors
    ///
    /// Returns an error if the import fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(&self.file)?;
        let mut entries: Vec<meme::model::MemoryEntry> = serde_json::from_str(&content)?;

        let count = entries.len();
        println!(
            "Importing {count} entries from {} (recomputing embeddings)...",
            self.file
        );

        let mut builder = meme::MemeBuilder::new();
        if let Some(uid) = &self.user_id {
            builder = builder.user_id(uid);
        }
        if let Some(sid) = &self.session_id {
            builder = builder.session_id(sid);
        }
        let meme = builder.build().await.map_err(|e| anyhow::anyhow!("{e}"))?;
        meme.import_entries(&mut entries)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("Imported {count} entries successfully.");
        Ok(())
    }
}
