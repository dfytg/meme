//! `meme add` — add dialogues or raw facts to the memory system.

use clap::Args;

use super::Context;

/// Add a dialogue or raw fact to the memory system.
///
/// Without `--speaker`, stores content as a direct fact via `put()`.
/// With `--speaker`, stores as a dialogue turn via `add()` + `flush()`.
#[derive(Debug, Args)]
pub(crate) struct AddCmd {
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
}

impl AddCmd {
    /// Execute the add command.
    ///
    /// # Errors
    ///
    /// Returns an error if adding fails.
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        if let Some(file_path) = &self.file {
            self.import_file(ctx, file_path).await
        } else {
            self.add_single(ctx).await
        }
    }

    /// Add a single dialogue turn or raw fact.
    async fn add_single(&self, ctx: &Context) -> anyhow::Result<()> {
        let content = self
            .content
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("content is required"))?;

        let meme = ctx.build_meme().await?;

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

            let mut d = meme::Dialogue::new(speaker, content);
            if let Some(ts) = timestamp {
                d = d.with_timestamp(ts);
            }
            meme.add(&[d]).await?;
            meme.flush().await?;
            println!("Added dialogue from {speaker}");
        } else {
            meme.put(content).await?;
            println!("Added fact");
        }
        Ok(())
    }

    /// Bulk-import dialogues from a text file ("speaker: text" per line).
    async fn import_file(&self, ctx: &Context, path: &str) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let mut dialogues = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("invalid JSONL at line {}: {e}", line_num + 1))?;

            let speaker = v
                .get("speaker")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_owned();
            let text = v
                .get("content")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_owned();
            let timestamp = v.get("timestamp").and_then(|s| s.as_str()).and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .ok()
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            });

            let mut d = meme::Dialogue::new(speaker, text);
            if let Some(ts) = timestamp {
                d = d.with_timestamp(ts);
            }
            dialogues.push(d);
        }

        let count = dialogues.len();
        let meme = ctx.build_meme().await?;
        meme.add(&dialogues).await?;
        meme.flush().await?;

        println!("Imported {count} dialogues from {path}");
        Ok(())
    }
}
