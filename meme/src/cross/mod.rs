//! Cross-session memory system — persistent memory across conversations.
//!
//! Provides session lifecycle management, event collection, context injection,
//! and memory consolidation (decay/merge/prune).

mod collector;
mod consolidation;
mod extractor;
mod injector;
mod orchestrator;

pub use collector::{EventCollector, RedactionFilter};
pub use consolidation::{
    ConsolidationActions, ConsolidationPolicy, ConsolidationStats, ConsolidationWorker,
};
pub use extractor::ObservationExtractor;
pub use injector::{ContextInjector, render_for_system_prompt};
pub use orchestrator::CrossOrchestrator;
