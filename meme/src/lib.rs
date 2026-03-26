//! # meme
//!
//! Long-term memory for AI agents.
//!
//! A Rust implementation of a production-grade memory pipeline:
//! 1. **Semantic Structured Compression** — dialogues → compact memory entries
//! 2. **Lifecycle Reconciliation** — LLM-driven ADD/UPDATE/DELETE/NOOP
//! 3. **Intent-Aware Retrieval Planning** — multi-view hybrid retrieval
//!
//! Memory is persistent across sessions — the vector store is stored on disk.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use meme::{Dialogue, Meme, MemeBuilder};
//!
//! # async fn example() -> meme::error::Result<()> {
//! let meme = MemeBuilder::new()
//!     .api_key("sk-...")
//!     .model("gpt-4.1-mini")
//!     .build()
//!     .await?;
//!
//! // Add a conversation
//! meme.add(&[
//!     Dialogue::new("Alice", "Let's meet at 2pm tomorrow"),
//!     Dialogue::new("Bob", "Sure, see you at Shibuya station"),
//! ]).await?;
//! meme.flush().await?;
//!
//! // Store a fact directly
//! meme.put("Alice prefers coffee over tea").await?;
//!
//! // Search & Q&A
//! let results = meme.search("Alice meeting").await?;
//! let answer = meme.ask("When will Alice meet?").await?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]

mod builder;
pub mod config;
pub(crate) mod embedding;
pub mod error;
pub(crate) mod llm;
mod meme;
pub mod model;
pub(crate) mod pipeline;
pub mod store;

pub use builder::MemeBuilder;
pub use meme::Meme;
pub use model::{Dialogue, Event, EventType, Memory, Scope};
pub use store::ConsolidationStats;
