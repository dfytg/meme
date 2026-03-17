//! LLM client abstraction — OpenAI-compatible async interface.

pub mod client;
pub mod prompt;
pub mod schema;

pub use client::{ChatOptions, LlmClient, Message, Role};
pub use schema::*;
