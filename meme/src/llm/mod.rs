//! LLM client abstraction — OpenAI-compatible async interface.

pub mod client;
pub mod json;
pub mod prompt;
pub mod schema;

pub use client::{ChatOptions, LlmClient, Message};
pub use schema::{
    AnswerResponse, CompletenessResponse, ExtractedEntry, ExtractionResponse,
    MissingQueriesResponse, QueryPlan, ReExtractResponse, ReconcileResponse,
};
