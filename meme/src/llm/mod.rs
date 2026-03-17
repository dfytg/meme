//! LLM client abstraction — OpenAI-compatible async interface.

pub mod client;
pub mod json;
pub mod prompt;
pub mod schema;

pub use client::{ChatOptions, LlmClient, Message, Role};
pub use json::extract_json_from_text;
pub use schema::{
    AnswerResponse, CompletenessResponse, ExtractedEntry, ExtractionResponse,
    MissingQueriesResponse, QueryPlan, ReExtractResponse, ReconcileResponse,
};
