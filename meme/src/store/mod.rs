//! Storage layer — vector store and relational store abstractions.

mod relational;
mod vector;

pub use relational::SqliteStore;
pub use vector::VectorStore;
