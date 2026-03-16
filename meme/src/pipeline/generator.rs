//! Answer Generator — synthesizes answers from retrieved memory contexts.

use std::sync::Arc;

use crate::error::Result;
use crate::llm::client::{ChatOptions, LlmClient, Message};
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

        let max_retries = 3;
        for attempt in 0..max_retries {
            match self.llm.chat(&messages, &opts).await {
                Ok(response) => match self.llm.extract_json(&response) {
                    Ok(result) => {
                        return Ok(result["answer"]
                            .as_str()
                            .unwrap_or(response.trim())
                            .to_owned());
                    }
                    Err(e) => {
                        if attempt + 1 < max_retries {
                            tracing::warn!(attempt = attempt + 1, error = %e, "answer parse failed, retrying");
                        } else {
                            return Ok(response.trim().to_owned());
                        }
                    }
                },
                Err(e) => {
                    if attempt + 1 < max_retries {
                        tracing::warn!(attempt = attempt + 1, error = %e, "answer generation failed, retrying");
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok("Failed to generate answer".to_owned())
    }
}
