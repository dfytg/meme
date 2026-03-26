//! Batch import — load a conversation from a JSONL file into memory.
//!
//! ```sh
//! MEME_LLM_API_KEY=sk-... cargo run --example batch_import
//! ```

use meme::model::Dialogue;

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

    // Simulate a multi-turn conversation.
    let dialogues = vec![
        Dialogue::new("PM", "Sprint review is on Friday at 10am in Room 3B."),
        Dialogue::new(
            "Dev",
            "I'll demo the new search feature and the billing fix.",
        ),
        Dialogue::new("PM", "Great. Also invite the QA team from Berlin."),
        Dialogue::new(
            "Dev",
            "Will do. Should I prepare slides for the Stripe integration?",
        ),
        Dialogue::new(
            "PM",
            "Yes, keep it under 5 minutes. Focus on the webhook handler.",
        ),
    ];

    let count = dialogues.len();
    meme.add_dialogues(dialogues).await?;
    meme.flush().await?;

    println!(
        "Imported {count} dialogues → {} memories\n",
        meme.count().await?
    );

    let answer = meme.ask("When and where is the sprint review?").await?;
    println!("Q: When and where is the sprint review?\nA: {answer}");

    Ok(())
}
