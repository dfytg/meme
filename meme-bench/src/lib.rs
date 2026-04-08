//! LOCOMO benchmark library for the meme memory system.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::future_not_send,
    reason = "benchmark binary uses stdout/stderr and single-threaded tokio"
)]

use tokio as _;
use tracing as _;
use tracing_subscriber as _;

pub mod dataset;
pub mod metrics;
pub mod runner;
