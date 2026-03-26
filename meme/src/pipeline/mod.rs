//! Three-stage memory pipeline: compression, synthesis, and retrieval.

mod extractor;
pub mod generator;
pub mod reconciler;
mod retriever;

pub use extractor::Extractor;
pub use retriever::HybridRetriever;
