//! `meme search` — semantic search over stored memories.

use clap::Args;
use comfy_table::{Cell, Table};

/// Search memories by semantic similarity.
#[derive(Debug, Args)]
pub struct SearchCmd {
    /// Search query text.
    pub query: String,

    /// Maximum number of results.
    #[arg(short, long, default_value_t = 10)]
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

impl SearchCmd {
    /// Execute the search command.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let meme = super::build_meme(self.user_id.as_deref(), self.session_id.as_deref()).await?;

        let entries = meme.search(&self.query).await?;

        let shown: Vec<_> = entries.iter().take(self.limit).collect();

        if self.json {
            println!("{}", serde_json::to_string_pretty(&shown)?);
        } else {
            if shown.is_empty() {
                println!("No results found.");
                return Ok(());
            }
            let mut table = Table::new();
            table.set_header(vec!["#", "ID", "Restatement", "Persons", "Topic"]);

            for (i, entry) in shown.iter().enumerate() {
                let restatement = super::truncate_str(&entry.restatement, 60);
                table.add_row(vec![
                    Cell::new(i + 1),
                    Cell::new(&entry.id.to_string()[..8]),
                    Cell::new(restatement),
                    Cell::new(entry.persons.join(", ")),
                    Cell::new(entry.topic.as_deref().unwrap_or("")),
                ]);
            }
            println!("{table}");
            println!("{} result(s)", shown.len());
        }

        Ok(())
    }
}
