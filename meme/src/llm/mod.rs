//! LLM client abstraction — OpenAI-compatible async interface.

mod client;
mod json;
pub mod prompt;
mod schema;

pub use client::{ChatOptions, LlmClient, Message};
pub use schema::{
    AnswerResponse, CompletenessResponse, ExtractedEntry, ExtractionResponse,
    MissingQueriesResponse, QueryPlan, ReExtractResponse, ReconcileResponse,
};
