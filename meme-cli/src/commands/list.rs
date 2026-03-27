//! `meme list` — list stored memory entries.

use clap::Args;
use comfy_table::{Cell, Table};

use super::Context;

/// List stored memory entries.
#[derive(Debug, Args)]
pub struct ListCmd {
    /// Maximum number of entries to show.
    #[arg(short, long, default_value_t = 20)]
    pub limit: usize,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl ListCmd {
    /// Execute the list command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let meme = ctx.build_meme().await?;

        let entries = meme.list().await?;

        let total = entries.len();

        if self.json {
            let limited: Vec<_> = entries.iter().take(self.limit).collect();
            println!("{}", serde_json::to_string_pretty(&limited)?);
        } else {
            let mut table = Table::new();
            table.set_header(vec!["#", "ID", "Content", "Persons", "Time", "Topic"]);

            for (i, entry) in entries.iter().take(self.limit).enumerate() {
                let content = super::truncate_str(&entry.content, 80);

                table.add_row(vec![
                    Cell::new(i + 1),
                    Cell::new(&entry.id.to_string()[..8]),
                    Cell::new(content),
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
