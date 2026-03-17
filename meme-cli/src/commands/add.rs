//! `meme add` — add dialogues to the memory system.

use clap::Args;

/// Add dialogues to the memory system.
#[derive(Debug, Args)]
pub struct AddCmd {
    /// Speaker name.
    #[arg(short, long)]
    pub speaker: Option<String>,

    /// Dialogue content (positional).
    pub content: Option<String>,

    /// Timestamp in ISO 8601 format.
    #[arg(short, long)]
    pub timestamp: Option<String>,

    /// Import dialogues from a JSONL file.
    #[arg(long, value_name = "FILE")]
    pub file: Option<String>,
}

impl AddCmd {
    /// Execute the add command.
    ///
    /// # Errors
    ///
    /// Returns an error if adding dialogues fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.file {
            self.import_file(file_path).await
        } else {
            self.add_single().await
        }
    }

    async fn add_single(&self) -> anyhow::Result<()> {
        let speaker = self.speaker.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--speaker is required when adding a single dialogue")
        })?;
        let content = self
            .content
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("content is required"))?;

        let timestamp = self
            .timestamp
            .as_deref()
            .map(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| anyhow::anyhow!("invalid timestamp: {e}"))
            })
            .transpose()?;

        let meme = meme::MemeBuilder::new()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        meme.add_dialogue(speaker, content, timestamp)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("Added dialogue from {speaker}");
        Ok(())
    }

    async fn import_file(&self, path: &str) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let mut dialogues = Vec::new();
        let mut id = 1u64;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("invalid JSONL at line {id}: {e}"))?;

            let speaker = v["speaker"].as_str().unwrap_or("unknown").to_owned();
            let content = v["content"].as_str().unwrap_or("").to_owned();
            let timestamp = v["timestamp"].as_str().and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .ok()
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            });

            let mut d = meme::model::Dialogue::new(speaker, content);
            if let Some(ts) = timestamp {
                d = d.with_timestamp(ts);
            }
            dialogues.push(d);
            id += 1;
        }

        let count = dialogues.len();
        let meme = meme::MemeBuilder::new()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        meme.add_dialogues(dialogues)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        meme.finalize().await.map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("Imported {count} dialogues from {path}");
        Ok(())
    }
}
