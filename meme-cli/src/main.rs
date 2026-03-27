//! meme-cli — interactive management tool for the meme memory system.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::future_not_send)]

mod commands;
mod config_loader;

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
    /// Add a dialogue or raw fact to the memory system.
    Add(commands::add::AddCmd),
    /// Ask a question against stored memories.
    Ask(commands::ask::AskCmd),
    /// Semantic search over stored memories.
    Search(commands::search::SearchCmd),
    /// Retrieve a single memory entry by UUID.
    Get(commands::get::GetCmd),
    /// Update a memory entry's content.
    Update(commands::update::UpdateCmd),
    /// Delete a memory entry by UUID.
    Delete(commands::delete::DeleteCmd),
    /// View the change history of a memory entry.
    History(commands::history::HistoryCmd),
    /// List stored memory entries.
    List(commands::list::ListCmd),
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

    if let Err(e) = rt.block_on(run(cli.command)) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(cmd: Command) -> anyhow::Result<()> {
    match cmd {
        Command::Init(c) => c.run(),
        Command::Add(c) => c.run().await,
        Command::Ask(c) => c.run().await,
        Command::Search(c) => c.run().await,
        Command::Get(c) => c.run().await,
        Command::Update(c) => c.run().await,
        Command::Delete(c) => c.run().await,
        Command::History(c) => c.run().await,
        Command::List(c) => c.run().await,
        Command::Export(c) => c.run().await,
        Command::Import(c) => c.run().await,
    }
}
