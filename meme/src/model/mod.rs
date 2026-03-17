//! Domain models — core data types for the memory system.

mod dialogue;
mod entry;
mod event;
mod filter;
mod scope;

pub use dialogue::Dialogue;
pub use entry::MemoryEntry;
pub use event::{EventType, MemoryEvent};
pub use filter::MetadataFilter;
pub use scope::Scope;
