//! Storage layer — `LanceDB` vector store with multi-view indexing and `SQLite` history tracking.

mod consolidation;
mod history;
mod vector;

pub use consolidation::ConsolidationStats;
pub use history::HistoryStore;
pub use vector::VectorStore;
