//! LLM client abstraction — OpenAI-compatible async interface.

mod client;
mod prompt;

pub use client::{ChatOptions, LlmClient, Message, OpenAiClient, Role};
pub use prompt::Prompts;
