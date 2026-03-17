//! LLM client abstraction — OpenAI-compatible async interface.

pub mod client;
pub mod prompt;

pub use client::{ChatOptions, LlmClient, Message, Role, extract_json_from_text};
pub use prompt::Prompts;
