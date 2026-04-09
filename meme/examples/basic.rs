//! Basic usage — add dialogues, then ask questions about them.
//!
//! ```sh
//! MEME_LLM_API_KEY=sk-... cargo run --example basic
//! ```

#![allow(
    unused_crate_dependencies,
    reason = "transitive lib deps are not directly used in this example"
)]

#[allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "example prints output to stdout/stderr"
)]
#[tokio::main]
async fn main() -> meme::error::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("meme=info")
        .init();

    let Ok(api_key) = std::env::var("MEME_LLM_API_KEY") else {
        eprintln!("set MEME_LLM_API_KEY env var to run this example");
        return Ok(());
    };

    let meme = meme::Meme::builder()
        .api_key(api_key)
        .model("gpt-4.1-mini")
        .clear_db(true)
        .build()
        .await?;

    meme.add(&[
        meme::Dialogue::new("Alice", "I'll be in Tokyo next Monday for the conference."),
        meme::Dialogue::new("Bob", "Great! Let's meet at Shibuya station at 3pm."),
        meme::Dialogue::new("Alice", "Sure. I'll bring the Q3 report for Acme Corp."),
    ])
    .await?;
    meme.flush().await?;

    println!("Stored {} memories\n", meme.count().await?);

    for q in [
        "Where will Alice and Bob meet?",
        "What will Alice bring?",
        "Which company is mentioned?",
    ] {
        let answer = meme.ask(q).await?;
        println!("Q: {q}\nA: {answer}\n");
    }

    Ok(())
}
