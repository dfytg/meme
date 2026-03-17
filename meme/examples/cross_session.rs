//! Cross-session memory — persist context across independent conversations.
//!
//! ```sh
//! MEME_LLM_API_KEY=sk-... cargo run --example cross_session
//! ```

use meme::config::Config;
use meme::cross::CrossOrchestrator;

#[allow(clippy::print_stdout)]
#[tokio::main]
async fn main() -> meme::error::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("meme=info")
        .init();

    let config = Config::load_default()?;
    let orch = CrossOrchestrator::new("my-project", &config)?;

    // Session 1: user asks to refactor auth module.
    let s1 = orch
        .start_session("conv-001", Some("Refactor the auth module"))
        .await?;
    println!("Session 1 started: {}", s1.memory_session_id);

    orch.record_message(
        &s1.memory_session_id,
        "user",
        "Please refactor auth to use JWT",
    )?;
    orch.record_message(
        &s1.memory_session_id,
        "assistant",
        "Done. Switched from session cookies to JWT tokens.",
    )?;

    let report = orch.stop_session(&s1.memory_session_id)?;
    println!(
        "Session 1 stopped: {} observations\n",
        report.observations_count
    );

    // Session 2: picks up context from session 1 automatically.
    let s2 = orch
        .start_session("conv-002", Some("Fix login bug"))
        .await?;
    println!("Session 2 context ({} chars):", s2.context_text.len());
    for line in s2.context_text.lines().take(5) {
        println!("  {line}");
    }

    let sessions = orch.list_sessions(10)?;
    println!("\n{} sessions recorded", sessions.len());

    Ok(())
}
