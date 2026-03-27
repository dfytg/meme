//! `meme history` — view change history for a memory entry.

use clap::Args;
use comfy_table::{Cell, Table};

use super::Context;

/// View the change history of a memory entry.
#[derive(Debug, Args)]
pub struct HistoryCmd {
    /// Memory entry UUID.
    pub id: String,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl HistoryCmd {
    /// Execute the history command.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let uuid = super::parse_uuid(&self.id)?;
        let meme = ctx.build_meme().await?;

        let events = meme.history(uuid).await?;

        if self.json {
            println!("{}", serde_json::to_string_pretty(&events)?);
        } else {
            if events.is_empty() {
                println!("No history found for {uuid}");
                return Ok(());
            }
            let mut table = Table::new();
            table.set_header(vec!["#", "Type", "Time", "Old", "New"]);

            for (i, event) in events.iter().enumerate() {
                let truncate = |s: &Option<String>| -> String {
                    s.as_deref()
                        .map(|v| super::truncate_str(v, 50))
                        .unwrap_or_default()
                };
                table.add_row(vec![
                    Cell::new(i + 1),
                    Cell::new(format!("{:?}", event.event_type)),
                    Cell::new(event.timestamp.format("%Y-%m-%d %H:%M:%S")),
                    Cell::new(truncate(&event.old_content)),
                    Cell::new(truncate(&event.new_content)),
                ]);
            }
            println!("{table}");
            println!("{} event(s)", events.len());
        }

        Ok(())
    }
}
