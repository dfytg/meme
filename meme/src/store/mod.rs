//! Storage layer — `LanceDB` vector store with multi-view indexing.

mod vector;

pub use vector::{ConsolidationStats, Scope, VectorStore};
