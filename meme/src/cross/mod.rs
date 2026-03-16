//! Cross-session memory system — persistent memory across conversations.
//!
//! Provides session lifecycle management, event collection, context injection,
//! and memory consolidation (decay/merge/prune).

mod collector;
mod consolidation;
mod injector;
mod orchestrator;
mod session;

pub use collector::EventCollector;
pub use consolidation::{ConsolidationPolicy, ConsolidationWorker};
pub use injector::ContextInjector;
pub use orchestrator::CrossOrchestrator;
pub use session::SessionManager;
