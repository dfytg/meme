//! LLM client abstraction — OpenAI-compatible async interface.

mod client;
mod json;
pub(crate) mod prompt;
mod schema;

pub(crate) use client::{ChatOptions, LlmClient, Message};
pub(crate) use schema::{
    AnswerResponse, CompletenessResponse, ExtractedEntry, ExtractionResponse,
    MissingQueriesResponse, QueryPlan, ReExtractResponse, ReconcileResponse,
};
