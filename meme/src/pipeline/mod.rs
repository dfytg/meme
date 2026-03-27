//! Three-stage memory pipeline: compression, reconciliation, and retrieval.

mod extractor;
mod generator;
mod reconciler;
mod retriever;

pub use extractor::Extractor;
pub use generator::generate;
pub use reconciler::reconcile;
pub use retriever::HybridRetriever;
