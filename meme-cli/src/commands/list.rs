//! `meme list` — list stored memory entries.

use clap::Args;
use comfy_table::{Cell, Table};

/// List stored memory entries.
#[derive(Debug, Args)]
pub struct ListCmd {
    /// Maximum number of entries to show.
    #[arg(short, long, default_value_t = 20)]
    pub limit: usize,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,

    /// User identifier for memory isolation.
    #[arg(long)]
    pub user_id: Option<String>,

    /// Session identifier for memory isolation.
    #[arg(long)]
    pub session_id: Option<String>,
}

impl ListCmd {
    /// Execute the list command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
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

        let total = entries.len();

        if self.json {
            let limited: Vec<_> = entries.iter().take(self.limit).collect();
            println!("{}", serde_json::to_string_pretty(&limited)?);
        } else {
            let mut table = Table::new();
            table.set_header(vec!["#", "Restatement", "Persons", "Time", "Topic"]);

            for (i, entry) in entries.iter().take(self.limit).enumerate() {
                let restatement = if entry.restatement.len() > 80 {
                    format!("{}...", &entry.restatement[..77])
                } else {
                    entry.restatement.clone()
                };

                table.add_row(vec![
                    Cell::new(i + 1),
                    Cell::new(restatement),
                    Cell::new(entry.persons.join(", ")),
                    Cell::new(
                        entry
                            .timestamp
                            .map(|ts| ts.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_default(),
                    ),
                    Cell::new(entry.topic.as_deref().unwrap_or("")),
                ]);
            }

            println!("{table}");
            if total > self.limit {
                println!(
                    "Showing {}/{total} entries (use --limit to show more)",
                    self.limit
                );
            } else {
                println!("{total} entries total");
            }
        }

        Ok(())
    }
}
