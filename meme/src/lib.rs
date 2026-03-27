//! # meme
//!
//! Long-term memory for AI agents.
//!
//! `meme` is a production-grade memory pipeline written in Rust.  It persists
//! knowledge extracted from conversations (or raw facts) to disk via
//! [LanceDB](https://lancedb.com) and tracks every change in a `SQLite` audit log.
//!
//! ## Pipeline
//!
//! 1. **Semantic Structured Compression** — dialogues are windowed and sent to
//!    an LLM which extracts atomic, self-contained [`Memory`] entries with
//!    structured metadata (timestamps, locations, persons, keywords, …).
//! 2. **Lifecycle Reconciliation** — each new entry is compared against existing
//!    memories in a single LLM call that decides ADD / UPDATE / DELETE / NOOP,
//!    preventing duplicates and resolving contradictions.
//! 3. **Intent-Aware Hybrid Retrieval** — queries are analyzed by an LLM to
//!    produce a retrieval plan that drives parallel semantic (ANN), lexical
//!    (FTS), and structured-metadata searches.  Optional reflection rounds
//!    iteratively refine coverage.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use meme::{Dialogue, Meme};
//!
//! # async fn example() -> meme::error::Result<()> {
//! let meme = Meme::builder()
//!     .api_key("sk-...")
//!     .model("gpt-4.1-mini")
//!     .build()
//!     .await?;
//!
//! // Ingest a conversation
//! meme.add(&[
//!     Dialogue::new("Alice", "Let's meet at 2pm tomorrow"),
//!     Dialogue::new("Bob", "Sure, see you at Shibuya station"),
//! ]).await?;
//! meme.flush().await?;
//!
//! // Store a fact directly (bypasses dialogue windowing)
//! meme.put("Alice prefers coffee over tea").await?;
//!
//! // Hybrid search & Q&A
//! let memories = meme.search("Alice meeting").await?;
//! let answer  = meme.ask("When will Alice meet?").await?;
//!
//! // CRUD
//! let all = meme.list().await?;
//! meme.update(all[0].id, "corrected content").await?;
//! meme.delete(all[0].id).await?;
//!
//! // Audit trail
//! let events = meme.history(all[0].id).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Public API
//!
//! | Entry point | Purpose |
//! |---|---|
//! | [`MemeBuilder`] | Fluent builder — configure API key, model, storage path |
//! | [`Meme`] | Runtime facade — `add`, `flush`, `put`, `search`, `ask`, CRUD, `consolidate` |
//! | [`Memory`] | A single self-contained unit of knowledge |
//! | [`Dialogue`] | Speaker + content input for conversation ingestion |
//! | [`Event`] / [`EventType`] | Change-history audit records |
//! | [`ConsolidationParams`] | Parameters for [`Meme::consolidate`] |
//! | [`ConsolidationStats`] | Summary returned by [`Meme::consolidate`] |
//!
//! ## Crate Layout
//!
//! | Module | Visibility | Contents |
//! |---|---|---|
//! | [`config`] | **pub** | Pure data configuration structs with validation |
//! | [`error`] | **pub** | [`Error`](error::Error) enum and [`Result`](error::Result) alias |
//! | [`model`] | **pub** | Domain types ([`Memory`], [`Dialogue`], [`Event`], …) |
//! | [`store`] | **pub** | LanceDB vector store, SQLite history store, consolidation |
//! | `embedding` | pub(crate) | API and optional ONNX embedding providers |
//! | `llm` | pub(crate) | OpenAI-compatible LLM client, prompts, JSON schemas |
//! | `pipeline` | pub(crate) | Extractor, reconciler, hybrid retriever, answer generator |

#![allow(clippy::missing_fields_in_debug)]

mod builder;
pub mod config;
pub(crate) mod embedding;
pub mod error;
pub(crate) mod llm;
mod meme;
pub mod model;
pub(crate) mod pipeline;
#[cfg(feature = "onnx")]
pub(crate) mod reranking;
pub mod store;

pub use builder::MemeBuilder;
pub use meme::Meme;
pub use model::{Dialogue, Event, EventType, Memory};
pub use store::{ConsolidationParams, ConsolidationStats};
