//! LOCOMO benchmark runner for the meme memory system.
//!
//! Evaluates long-term conversational memory quality using the LOCOMO
//! benchmark format: feeds dialogues into meme, asks questions, and
//! computes per-category F1 scores.
//!
//! ## Usage
//!
//! ```sh
//! # Run with a LOCOMO-format JSON dataset
//! MEME_LLM_API_KEY=sk-... meme-bench run --dataset locomo10.json
//!
//! # Run with a custom benchmark file
//! MEME_LLM_API_KEY=sk-... meme-bench run --dataset my_bench.json --model gpt-4.1-mini
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::future_not_send)]

mod dataset;
mod metrics;
mod runner;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "meme-bench", version, about = "LOCOMO benchmark for meme")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the benchmark against a LOCOMO-format dataset.
    Run(runner::RunCmd),
    /// Generate a sample benchmark dataset for testing.
    Sample(SampleCmd),
}

#[derive(Debug, clap::Args)]
struct SampleCmd {
    /// Output file path.
    #[arg(short, long, default_value = "sample_bench.json")]
    output: String,
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
        Command::Run(cmd) => rt.block_on(cmd.run()),
        Command::Sample(cmd) => generate_sample(&cmd.output),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn generate_sample(path: &str) -> anyhow::Result<()> {
    let sample = dataset::sample_dataset();
    let json = serde_json::to_string_pretty(&sample)?;
    std::fs::write(path, json)?;
    println!("Sample benchmark written to {path}");
    Ok(())
}
