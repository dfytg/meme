//! `meme config` — show effective configuration.

use clap::Args;

use super::Context;
use crate::config_loader;

/// Show the effective configuration.
#[derive(Debug, Args)]
pub struct ConfigCmd;

impl ConfigCmd {
    /// Execute the config command.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[allow(clippy::unused_self)]
    pub fn run(&self, ctx: &Context) -> anyhow::Result<()> {
        let config = match &ctx.config_path {
            Some(path) => config_loader::from_file(path)?,
            None => config_loader::load_default(),
        };

        let default_path = config_loader::default_config_path();
        let path = ctx.config_path.as_deref().unwrap_or(&default_path);
        let exists = path.exists();

        println!(
            "Config file: {} {}",
            path.display(),
            if exists {
                "(found)"
            } else {
                "(not found, using defaults)"
            }
        );
        if let Some(ns) = &ctx.namespace {
            println!("Namespace:   {ns}");
        }
        println!();
        println!("{}", toml::to_string_pretty(&config)?);
        Ok(())
    }
}
