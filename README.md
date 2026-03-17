# meme

[![CI][ci-badge]][ci-url]
[![License][license-badge]][license-url]
[![Rust][rust-badge]][rust-url]

[ci-badge]: https://github.com/qntx/meme/actions/workflows/rust.yml/badge.svg
[ci-url]: https://github.com/qntx/meme/actions/workflows/rust.yml
[license-badge]: https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg
[license-url]: LICENSE-MIT
[rust-badge]: https://img.shields.io/badge/rust-edition%202024-orange.svg
[rust-url]: https://doc.rust-lang.org/edition-guide/

**High-performance long-term memory for AI agents — three-stage pipeline with semantic compression, hybrid retrieval, and persistent vector storage, written in Rust.**

meme implements the [SimpleMem](3rdparty/SimpleMem/) three-stage memory pipeline with a production-grade Rust core: (1) **Semantic Structured Compression** extracts lossless, disambiguated memory entries from dialogues via LLM, (2) **Online Semantic Synthesis** deduplicates at write time, and (3) **Intent-Aware Retrieval Planning** combines semantic, lexical (FTS), and structured metadata search with LLM-driven reflection. Memory is stored persistently on disk via LanceDB.

## Crates

| Crate | | Description |
| --- | --- | --- |
| **[`meme`](meme/)** | [![crates.io][meme-crate]][meme-crate-url] [![docs.rs][meme-doc]][meme-doc-url] | Core library — pipeline, vector store, embedding, LLM client |
| **[`meme-cli`](meme-cli/)** | [![crates.io][cli-crate]][cli-crate-url] | CLI tool — add dialogues, ask questions, export memories |

[meme-crate]: https://img.shields.io/crates/v/meme.svg
[meme-crate-url]: https://crates.io/crates/meme
[meme-doc]: https://img.shields.io/docsrs/meme.svg
[meme-doc-url]: https://docs.rs/meme
[cli-crate]: https://img.shields.io/crates/v/meme-cli.svg
[cli-crate-url]: https://crates.io/crates/meme-cli

## Quick Start

### Install the CLI

**Shell** (macOS / Linux):

```sh
curl -fsSL https://sh.qntx.fun/meme | sh
```

**PowerShell** (Windows):

```powershell
irm https://sh.qntx.fun/meme/ps | iex
```

### CLI

```bash
# Initialize configuration
meme init

# Add dialogues
meme add -s Alice "I'll be in Tokyo next Monday for the conference."
meme add -s Bob "Let's meet at Shibuya station at 3pm."

# Import from JSONL file
meme add --file conversation.jsonl

# Ask questions
meme ask "Where will Alice and Bob meet?"

# List stored memories
meme list
meme list --json

# Export / import
meme export -o memories.json
```

### Library

```rust
use meme::MemeBuilder;

let meme = MemeBuilder::new()
    .api_key("sk-...")
    .model("gpt-4.1-mini")
    .build()
    .await?;

// Add dialogues — automatically extracted into structured memory entries.
meme.add_dialogue("Alice", "Let's meet at 2pm tomorrow", None).await?;
meme.add_dialogue("Bob", "Sure, I'll bring the Q3 report", None).await?;
meme.finalize().await?;

// Ask questions — hybrid retrieval + LLM answer generation.
let answer = meme.ask("When will Alice meet?").await?;
```

See [`examples/`](meme/examples/) for more: [basic](meme/examples/basic.rs), [batch import](meme/examples/batch_import.rs).

## Feature Flags

| Feature | Description |
| --- | --- |
| `api-embedding` | Remote API-based embedding (enabled by default) |
| `onnx` | Local ONNX Runtime embedding via `ort` + `tokenizers` |

## Configuration

Configuration is loaded from `~/.meme/config.toml` with environment variable overrides:

| Env Var | Description |
| --- | --- |
| `MEME_LLM_API_KEY` | OpenAI-compatible API key |
| `MEME_LLM_BASE_URL` | API base URL |
| `MEME_LLM_MODEL` | Model name (default: `gpt-4.1-mini`) |
| `MEME_EMBEDDING_PROVIDER` | `api` or `onnx` |

## Architecture

```text
Dialogues ──► MemoryBuilder ──► VectorStore (LanceDB)
               (Stage 1+2)         │
                                   ├─ Semantic search (dense vectors)
Query ──► HybridRetriever ────────►├─ Keyword search (FTS / Tantivy)
           (Stage 3)               ├─ Structured search (metadata filters)
               │                   │
               ▼                   │
          LLM Planning ◄───────────┘
           + Reflection
               │
               ▼
          Answer Generation
```

- **`meme`** — Core library. `Meme` facade wraps the three-stage pipeline behind `add_dialogue()` / `ask()`. `VectorStore` uses LanceDB for embedded vector + FTS indexing. `Embedder` supports API and local ONNX backends via enum dispatch (zero-cost). `LlmClient` is an OpenAI-compatible HTTP client with retry + exponential backoff.
- **`meme-cli`** — Interactive CLI. TOML + env var configuration, JSONL import, table/JSON output.

## Three-Stage Pipeline

### Stage 1: Semantic Structured Compression

Dialogues are windowed and sent to an LLM to extract **atomic, self-contained memory entries**. Each entry contains:

- **Lossless restatement** — complete sentence with no pronouns, no relative time
- **Keywords** — for BM25-style lexical matching
- **Structured metadata** — timestamp, location, persons, entities, topic

### Stage 2: Online Semantic Synthesis

Previous-window entries are passed as context during extraction to avoid duplicating information across overlapping windows.

### Stage 3: Intent-Aware Retrieval Planning

A single LLM call analyzes the query to produce a **unified plan** containing:

- Extracted keywords, persons, entities, time expressions
- Targeted search queries for semantic retrieval
- Required information types for completeness assessment

The plan drives parallel execution of three search views (semantic, keyword, structured), followed by optional **reflection** rounds that assess completeness and issue additional targeted queries.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.

---

<div align="center">

A **[QNTX](https://qntx.fun)** open-source project.

<a href="https://qntx.fun"><img alt="QNTX" width="369" src="https://raw.githubusercontent.com/qntx/.github/main/profile/qntx-banner.svg" /></a>

<!--prettier-ignore-->
Code is law. We write both.

</div>
