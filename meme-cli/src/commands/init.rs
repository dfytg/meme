//! `meme init` — initialize configuration and database.

use clap::Args;
use meme::config::{Config, default_config_path};

/// Initialize meme configuration and database.
#[derive(Debug, Args)]
pub struct InitCmd {
    /// Force overwrite existing configuration.
    #[arg(long)]
    force: bool,
}

impl InitCmd {
    /// Execute the init command.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn run(&self) -> anyhow::Result<()> {
        let config_path = default_config_path();

        if config_path.exists() && !self.force {
            println!("Configuration already exists at: {}", config_path.display());
            println!("Use --force to overwrite.");
            return Ok(());
        }

        let config = Config::default();
        config.save(&config_path)?;

        println!("Configuration created at: {}", config_path.display());
        println!();
        println!("Next steps:");
        println!("  1. Edit the config file to set your API key:");
        println!("     {}", config_path.display());
        println!("  2. Or set the MEME_LLM_API_KEY environment variable");
        println!("  3. Run `meme add` to start adding dialogues");

        Ok(())
    }
}
