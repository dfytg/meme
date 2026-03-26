//! Storage layer — `LanceDB` vector store with multi-view indexing and `SQLite` history tracking.

mod history;
mod vector;

pub use history::HistoryStore;
pub use vector::VectorStore;
