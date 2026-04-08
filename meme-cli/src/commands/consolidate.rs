//! `meme consolidate` — consolidate memories (decay, merge, prune).

use clap::Args;
use meme::store::ConsolidationParams;

use super::Context;

/// Consolidate memories: decay old entries, merge near-duplicates, prune low-importance.
#[derive(Debug, Args)]
pub(crate) struct ConsolidateCmd {
    /// Maximum age in days before decay applies.
    #[arg(long, default_value_t = 90)]
    pub max_age_days: u32,

    /// Decay factor (0.0–1.0) applied to old entries' importance.
    #[arg(long, default_value_t = 0.95)]
    pub decay_factor: f64,

    /// Cosine similarity threshold for merging near-duplicates.
    #[arg(long, default_value_t = 0.95)]
    pub merge_threshold: f64,

    /// Minimum importance score to keep an entry.
    #[arg(long, default_value_t = 0.1)]
    pub min_importance: f64,
}

impl ConsolidateCmd {
    /// Execute the consolidate command.
    ///
    /// # Errors
    ///
    /// Returns an error if consolidation fails.
    #[allow(
        clippy::print_stdout,
        reason = "CLI command prints consolidation stats to stdout"
    )]
    pub(crate) async fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let meme = ctx.build_meme().await?;
        let params = ConsolidationParams {
            max_age_days: self.max_age_days,
            decay_factor: self.decay_factor,
            merge_threshold: self.merge_threshold,
            min_importance: self.min_importance,
        };
        let stats = meme.consolidate(&params).await?;

        println!("Consolidation complete ({:.1}s):", stats.duration_secs);
        println!("  Scanned: {}", stats.scanned);
        println!("  Decayed: {}", stats.decayed);
        println!("  Merged:  {}", stats.merged);
        println!("  Pruned:  {}", stats.pruned);
        Ok(())
    }
}
