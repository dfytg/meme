//! `meme search` — semantic search over stored memories.

use clap::Args;
use comfy_table::{Cell, Table};

use super::Context;

/// Search memories by semantic similarity.
#[derive(Debug, Args)]
pub(crate) struct SearchCmd {
    /// Search query text.
    pub query: String,

    /// Maximum number of results.
    #[arg(short, long, default_value_t = 10)]
    pub limit: usize,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl SearchCmd {
    /// Execute the search command.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let meme = ctx.build_meme().await?;

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
            table.set_header(vec!["#", "ID", "Content", "Persons", "Topic"]);

            for (i, entry) in shown.iter().enumerate() {
                let content = super::truncate_str(&entry.content, 60);
                table.add_row(vec![
                    Cell::new(i + 1),
                    Cell::new(&entry.id.to_string()[..8]),
                    Cell::new(content),
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
