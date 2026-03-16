//! `meme session` — manage cross-session memory.

use clap::{Args, Subcommand};
use comfy_table::{Cell, Table};
use meme::config::Config;
use meme::cross::CrossOrchestrator;

/// Manage cross-session memory.
#[derive(Debug, Args)]
pub struct SessionCmd {
    #[command(subcommand)]
    pub action: SessionAction,
}

/// Session subcommands.
#[derive(Debug, Subcommand)]
pub enum SessionAction {
    /// Start a new session.
    Start {
        /// External session identifier.
        id: String,
        /// Project name.
        #[arg(short, long, default_value = "default")]
        project: String,
        /// User prompt / request.
        #[arg(short = 'm', long)]
        prompt: Option<String>,
    },
    /// Stop (finalize) an active session.
    Stop {
        /// Memory session ID (UUID).
        id: String,
        /// Project name.
        #[arg(short, long, default_value = "default")]
        project: String,
    },
    /// List recent sessions.
    List {
        /// Project name.
        #[arg(short, long, default_value = "default")]
        project: String,
        /// Maximum number of sessions to show.
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },
}

impl SessionCmd {
    /// Execute the session command.
    ///
    /// # Errors
    ///
    /// Returns an error if the session operation fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let config = Config::load_default().map_err(|e| anyhow::anyhow!("{e}"))?;

        match &self.action {
            SessionAction::Start {
                id,
                project,
                prompt,
            } => {
                let orch =
                    CrossOrchestrator::new(project, &config).map_err(|e| anyhow::anyhow!("{e}"))?;
                let result = orch
                    .start_session(id, prompt.as_deref())
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                println!("Session started:");
                println!("  Memory Session ID: {}", result.memory_session_id);
                if result.context_text.is_empty() {
                    println!("  No previous context available.");
                } else {
                    println!("  Injected context ({} chars):", result.context_text.len());
                    for line in result.context_text.lines().take(5) {
                        println!("    {line}");
                    }
                    if result.context_text.lines().count() > 5 {
                        println!("    ...");
                    }
                }
            }
            SessionAction::Stop { id, project } => {
                let session_id =
                    uuid::Uuid::parse_str(id).map_err(|e| anyhow::anyhow!("invalid UUID: {e}"))?;
                let orch =
                    CrossOrchestrator::new(project, &config).map_err(|e| anyhow::anyhow!("{e}"))?;
                let report = orch
                    .stop_session(&session_id)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                println!("Session stopped:");
                println!("  Observations: {}", report.observations_count);
                println!("  Summary generated: {}", report.summary_generated);
                println!("  Entries stored: {}", report.entries_stored);
            }
            SessionAction::List { project, limit } => {
                let orch =
                    CrossOrchestrator::new(project, &config).map_err(|e| anyhow::anyhow!("{e}"))?;
                let sessions = orch
                    .list_sessions(*limit)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                if sessions.is_empty() {
                    println!("No sessions found for project '{project}'.");
                    return Ok(());
                }

                let mut table = Table::new();
                table.set_header(vec!["Session ID", "Status", "Started", "Prompt"]);

                for s in &sessions {
                    table.add_row(vec![
                        Cell::new(s.memory_session_id),
                        Cell::new(format!("{:?}", s.status)),
                        Cell::new(s.started_at.format("%Y-%m-%d %H:%M")),
                        Cell::new(
                            s.user_prompt
                                .as_deref()
                                .map(|p| {
                                    if p.len() > 50 {
                                        let boundary = p.floor_char_boundary(47);
                                        format!("{}...", &p[..boundary])
                                    } else {
                                        p.to_owned()
                                    }
                                })
                                .unwrap_or_default(),
                        ),
                    ]);
                }

                println!("{table}");
            }
        }

        Ok(())
    }
}
