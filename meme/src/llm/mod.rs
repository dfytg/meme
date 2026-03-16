//! LLM client abstraction — OpenAI-compatible async interface.

pub mod client;
pub mod prompt;

pub use client::{ChatOptions, LlmClient, Message, OpenAiClient, Role};
pub use prompt::Prompts;
