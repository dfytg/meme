//! Three-stage memory pipeline: compression, synthesis, and retrieval.

mod builder;
mod generator;
mod retriever;

pub use builder::MemoryBuilder;
pub use retriever::HybridRetriever;
