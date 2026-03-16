//! Answer Generator — synthesizes answers from retrieved memory contexts.

use std::sync::Arc;

use crate::error::Result;
use crate::llm::client::{ChatOptions, LlmClient, Message, extract_json_from_text};
use crate::llm::prompt::Prompts;
use crate::model::MemoryEntry;

/// Generates concise answers from retrieved memory entries.
pub struct AnswerGenerator {
    llm: Arc<dyn LlmClient>,
}

impl std::fmt::Debug for AnswerGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnswerGenerator").finish_non_exhaustive()
    }
}

impl AnswerGenerator {
    /// Create a new answer generator.
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// Generate an answer for a query given retrieved contexts.
    ///
    /// # Errors
    ///
    /// Returns an error if the LLM call fails.
    pub async fn generate(&self, query: &str, contexts: &[MemoryEntry]) -> Result<String> {
        if contexts.is_empty() {
            return Ok("No relevant information found".to_owned());
        }

        let context_str = Prompts::format_contexts(contexts);
        let prompt = Prompts::answer(query, &context_str);

        let messages = vec![
            Message::system(
                "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.",
            ),
            Message::user(prompt),
        ];
        let opts = ChatOptions {
            temperature: 0.1,
            json_mode: false,
        };

        let parse_retries = 2;
        for attempt in 0..=parse_retries {
            let response = match self.llm.chat(&messages, &opts).await {
                Ok(r) => r,
                Err(e) => return Err(e),
            };
            match extract_json_from_text(&response) {
                Ok(result) => {
                    return Ok(result["answer"]
                        .as_str()
                        .unwrap_or(response.trim())
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
}
