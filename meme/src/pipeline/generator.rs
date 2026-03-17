//! Answer generation — synthesizes answers from retrieved memory contexts.

use crate::config::PipelineConfig;
use crate::error::Result;
use crate::llm::{AnswerResponse, ChatOptions, LlmClient, Message, prompt};
use crate::model::MemoryEntry;

/// Generate an answer for a query given retrieved contexts.
///
/// If `pipeline_cfg.custom_answer_prompt` is set, it replaces the built-in prompt.
///
/// # Errors
///
/// Returns an error if the LLM call fails.
#[tracing::instrument(skip(llm, contexts, pipeline_cfg), fields(contexts = contexts.len()))]
pub async fn generate(
    llm: &LlmClient,
    query: &str,
    contexts: &[MemoryEntry],
    pipeline_cfg: &PipelineConfig,
) -> Result<String> {
    if contexts.is_empty() {
        return Ok("No relevant information found".to_owned());
    }

    let mut sorted: Vec<&MemoryEntry> = contexts.iter().collect();
    sorted.sort_by_key(|e| e.timestamp);
    let context_str = prompt::format_contexts(&sorted);

    #[allow(clippy::literal_string_with_formatting_args)]
    let user_prompt = pipeline_cfg.custom_answer_prompt.as_ref().map_or_else(
        || prompt::answer(query, &context_str),
        |custom| {
            custom
                .replace("{query}", query)
                .replace("{context}", &context_str)
        },
    );

    let messages = vec![
        Message::system("You are a professional Q&A assistant. Output valid JSON."),
        Message::user(user_prompt),
    ];
    let opts = ChatOptions {
        temperature: 0.1,
        json_mode: true,
    };

    let resp: AnswerResponse = llm.chat_structured(&messages, &opts).await?;
    if resp.answer.is_empty() {
        return Ok("No relevant information found".to_owned());
    }
    Ok(resp.answer)
}
