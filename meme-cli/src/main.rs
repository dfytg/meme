//! meme-cli — interactive management tool for the meme memory system.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::future_not_send)]

mod commands;
mod config_loader;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Long-term memory for AI agents.
#[derive(Debug, Parser)]
#[command(name = "meme", version, about)]
struct Cli {
    /// Namespace for memory isolation (opaque, caller-defined).
    #[arg(long, short = 'n', global = true)]
    namespace: Option<String>,

    /// Path to configuration file (default: ~/.meme/config.toml).
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

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
    /// Count stored memory entries.
    Count(commands::count::CountCmd),
    /// Clear all stored memories.
    Clear(commands::clear::ClearCmd),
    /// Consolidate memories (decay, merge, prune).
    Consolidate(commands::consolidate::ConsolidateCmd),
    /// Export memory entries to JSON.
    Export(commands::data::ExportCmd),
    /// Import memory entries from a JSON file.
    Import(commands::data::ImportCmd),
    /// Show effective configuration.
    Config(commands::config::ConfigCmd),
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

    let ctx = commands::Context {
        namespace: cli.namespace,
        config_path: cli.config,
    };

    if let Err(e) = rt.block_on(run(cli.command, &ctx)) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(cmd: Command, ctx: &commands::Context) -> anyhow::Result<()> {
    match cmd {
        Command::Init(c) => c.run(ctx),
        Command::Add(c) => c.run(ctx).await,
        Command::Ask(c) => c.run(ctx).await,
        Command::Search(c) => c.run(ctx).await,
        Command::Get(c) => c.run(ctx).await,
        Command::Update(c) => c.run(ctx).await,
        Command::Delete(c) => c.run(ctx).await,
        Command::History(c) => c.run(ctx).await,
        Command::List(c) => c.run(ctx).await,
        Command::Count(c) => c.run(ctx).await,
        Command::Clear(c) => c.run(ctx).await,
        Command::Consolidate(c) => c.run(ctx).await,
        Command::Export(c) => c.run(ctx).await,
        Command::Import(c) => c.run(ctx).await,
        Command::Config(c) => c.run(ctx),
    }
}
