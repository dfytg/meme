//! meme-cli — interactive management tool for the meme memory system.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::future_not_send)]

mod commands;

use clap::{Parser, Subcommand};

/// Long-term memory for AI agents.
#[derive(Debug, Parser)]
#[command(name = "meme", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Initialize configuration and database.
    Init(commands::init::InitCmd),
    /// Add dialogues to the memory system.
    Add(commands::add::AddCmd),
    /// Ask a question against stored memories.
    Ask(commands::ask::AskCmd),
    /// List stored memory entries.
    List(commands::list::ListCmd),
    /// Manage cross-session memory.
    Session(commands::session::SessionCmd),
    /// Manually trigger memory consolidation.
    Consolidate(commands::consolidate::ConsolidateCmd),
    /// Export memory entries to JSON.
    Export(commands::export::ExportCmd),
    /// Import memory entries from a JSON file.
    Import(commands::export::ImportCmd),
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let result = match cli.command {
        Command::Init(cmd) => cmd.run().map_err(|e| e.to_string()),
        Command::Add(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
        Command::Ask(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
        Command::List(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
        Command::Session(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
        Command::Consolidate(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
        Command::Export(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
        Command::Import(cmd) => rt.block_on(cmd.run()).map_err(|e| e.to_string()),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
