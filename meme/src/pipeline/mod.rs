//! Three-stage memory pipeline: compression, synthesis, and retrieval.

mod builder;
pub mod generator;
pub mod reconciler;
mod retriever;

pub use builder::Extractor;
pub use retriever::HybridRetriever;
