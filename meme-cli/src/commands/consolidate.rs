//! `meme consolidate` — manually trigger memory consolidation.

use clap::Args;
use meme::config::Config;
use meme::cross::CrossOrchestrator;

/// Manually trigger memory consolidation (decay/merge/prune).
#[derive(Debug, Args)]
pub struct ConsolidateCmd {
    /// Project name.
    #[arg(short, long, default_value = "default")]
    pub project: String,

    /// Dry run — show what would happen without making changes.
    #[arg(long)]
    pub dry_run: bool,
}

impl ConsolidateCmd {
    /// Execute the consolidate command.
    ///
    /// # Errors
    ///
    /// Returns an error if consolidation fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        if self.dry_run {
            println!("Dry run mode — no changes will be made.");
        }

        let config = Config::load_default().map_err(|e| anyhow::anyhow!("{e}"))?;
        let orch =
            CrossOrchestrator::new(&self.project, &config).map_err(|e| anyhow::anyhow!("{e}"))?;

        let stats = orch
            .consolidate()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("Consolidation complete:");
        println!("  Scanned:  {}", stats.scanned);
        println!("  Decayed:  {}", stats.decayed);
        println!("  Merged:   {}", stats.merged);
        println!("  Pruned:   {}", stats.pruned);

        Ok(())
    }
}
