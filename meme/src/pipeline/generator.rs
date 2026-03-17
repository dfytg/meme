//! Answer generation — synthesizes answers from retrieved memory contexts.

use crate::error::Result;
use crate::llm::{ChatOptions, LlmClient, Message, extract_json_from_text, prompt};
use crate::model::MemoryEntry;

/// Generate an answer for a query given retrieved contexts.
///
/// # Errors
///
/// Returns an error if the LLM call fails.
pub async fn generate(llm: &LlmClient, query: &str, contexts: &[MemoryEntry]) -> Result<String> {
    if contexts.is_empty() {
        return Ok("No relevant information found".to_owned());
    }

    let context_str = prompt::format_contexts(contexts);
    let prompt = prompt::answer(query, &context_str);

    let messages = vec![
        Message::system(
            "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.",
        ),
        Message::user(prompt),
    ];
    let opts = ChatOptions {
        temperature: 0.1,
        json_mode: true,
    };

    let parse_retries = 2;
    for attempt in 0..=parse_retries {
        let response = match llm.chat(&messages, &opts).await {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        match extract_json_from_text(&response) {
            Ok(result) => {
                return Ok(result["answer"]
                    .as_str()
                    .unwrap_or_else(|| response.trim())
                    .to_owned());
            }
            Err(e) => {
                if attempt < parse_retries {
                    tracing::warn!(attempt = attempt + 1, error = %e, "answer parse failed, retrying LLM");
                } else {
                    return Ok(response.trim().to_owned());
                }
            }
        }
    }

    Ok("Failed to generate answer".to_owned())
}
