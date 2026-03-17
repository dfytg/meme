//! Basic usage — add dialogues, then ask questions about them.
//!
//! ```sh
//! MEME_LLM_API_KEY=sk-... cargo run --example basic
//! ```

#[allow(clippy::print_stdout)]
#[tokio::main]
async fn main() -> meme::error::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("meme=info")
        .init();

    let meme = meme::MemeBuilder::new()
        .model("gpt-4.1-mini")
        .clear_db(true)
        .build()
        .await?;

    meme.add_dialogue(
        "Alice",
        "I'll be in Tokyo next Monday for the conference.",
        None,
    )
    .await?;
    meme.add_dialogue("Bob", "Great! Let's meet at Shibuya station at 3pm.", None)
        .await?;
    meme.add_dialogue(
        "Alice",
        "Sure. I'll bring the Q3 report for Acme Corp.",
        None,
    )
    .await?;
    meme.finalize().await?;

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
