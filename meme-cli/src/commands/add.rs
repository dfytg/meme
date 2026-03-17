//! `meme add` — add dialogues or raw facts to the memory system.

use clap::Args;

/// Add a dialogue or raw fact to the memory system.
///
/// Without `--speaker`, stores content as a direct fact (bypasses dialogue windowing).
/// With `--speaker`, stores as a dialogue turn.
#[derive(Debug, Args)]
pub struct AddCmd {
    /// Speaker name (omit for direct fact ingestion).
    #[arg(short, long)]
    pub speaker: Option<String>,

    /// Content text (positional).
    pub content: Option<String>,

    /// Timestamp in ISO 8601 format.
    #[arg(short, long)]
    pub timestamp: Option<String>,

    /// Import dialogues from a JSONL file.
    #[arg(long, value_name = "FILE")]
    pub file: Option<String>,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl AddCmd {
    /// Execute the add command.
    ///
    /// # Errors
    ///
    /// Returns an error if adding fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.file {
            self.import_file(file_path).await
        } else {
            self.add_single().await
        }
    }

    async fn add_single(&self) -> anyhow::Result<()> {
        let content = self
            .content
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("content is required"))?;

        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;

        if let Some(speaker) = &self.speaker {
            let timestamp = self
                .timestamp
                .as_deref()
                .map(|s| {
                    chrono::DateTime::parse_from_rfc3339(s)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .map_err(|e| anyhow::anyhow!("invalid timestamp: {e}"))
                })
                .transpose()?;

            meme.add_dialogue(speaker, content, timestamp).await?;
            println!("Added dialogue from {speaker}");
        } else {
            meme.add(content).await?;
            println!("Added fact");
        }
        Ok(())
    }

    async fn import_file(&self, path: &str) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let mut dialogues = Vec::new();
        let mut line_num = 1u64;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("invalid JSONL at line {line_num}: {e}"))?;

            let speaker = v["speaker"].as_str().unwrap_or("unknown").to_owned();
            let text = v["content"].as_str().unwrap_or("").to_owned();
            let timestamp = v["timestamp"].as_str().and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .ok()
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            });

            let mut d = meme::model::Dialogue::new(speaker, text);
            if let Some(ts) = timestamp {
                d = d.with_timestamp(ts);
            }
            dialogues.push(d);
            line_num += 1;
        }

        let count = dialogues.len();
        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;
        meme.add_dialogues(dialogues).await?;
        meme.finalize().await?;

        println!("Imported {count} dialogues from {path}");
        Ok(())
    }
}
