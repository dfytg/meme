//! Storage layer — `LanceDB` vector store with multi-view indexing and history tracking.

mod history;
mod vector;

pub use history::HistoryStore;
pub use vector::{ConsolidationStats, Scope, VectorStore};
