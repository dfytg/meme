//! Storage layer — `LanceDB` vector store with multi-view indexing and `SQLite` history tracking.

mod consolidation;
mod history;
mod vector;

pub use consolidation::{ConsolidationParams, ConsolidationStats, consolidate};
pub use history::HistoryStore;
pub use vector::VectorStore;
