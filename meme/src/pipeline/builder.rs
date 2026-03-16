//! Stage 1+2: Memory Builder — Semantic Structured Compression + Online Semantic Synthesis.
//!
//! Processes dialogue windows through an LLM to extract structured, atomic memory entries.
//! Supports parallel processing of multiple windows via tokio tasks.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::PipelineConfig;
use crate::error::{Error, Result};
use crate::llm::client::{ChatOptions, LlmClient, Message, extract_json_from_text};
use crate::llm::prompt::Prompts;
use crate::model::{Dialogue, MemoryEntry};

/// Memory builder that implements Stage 1 (Semantic Structured Compression)
/// and Stage 2 (Online Semantic Synthesis) of the `SimpleMem` pipeline.
pub struct MemoryBuilder {
    llm: Arc<dyn LlmClient>,
    window_size: usize,
    overlap_size: usize,
    step_size: usize,
    max_parallel_workers: usize,
    dialogue_buffer: Vec<Dialogue>,
    processed_count: usize,
    previous_entries: Vec<MemoryEntry>,
}

impl std::fmt::Debug for MemoryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryBuilder")
            .field("window_size", &self.window_size)
            .field("overlap_size", &self.overlap_size)
            .field("buffer_len", &self.dialogue_buffer.len())
            .field("processed_count", &self.processed_count)
            .finish()
    }
}

impl MemoryBuilder {
    /// Create a new memory builder.
    pub fn new(
        llm: Arc<dyn LlmClient>,
        pipeline_cfg: &PipelineConfig,
        max_parallel_workers: usize,
    ) -> Self {
        let step_size = pipeline_cfg
            .window_size
            .saturating_sub(pipeline_cfg.overlap_size)
            .max(1);
        Self {
            llm,
            window_size: pipeline_cfg.window_size,
            overlap_size: pipeline_cfg.overlap_size,
            step_size,
            max_parallel_workers,
            dialogue_buffer: Vec::new(),
            processed_count: 0,
            previous_entries: Vec::new(),
        }
    }

    /// Add a single dialogue to the buffer.
    ///
    /// Returns extracted entries when a full window has been processed.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction fails.
    pub async fn add_dialogue(&mut self, dialogue: Dialogue) -> Result<Vec<MemoryEntry>> {
        self.dialogue_buffer.push(dialogue);
        if self.dialogue_buffer.len() >= self.window_size {
            return self.process_window().await;
        }
        Ok(Vec::new())
    }

    /// Batch add dialogues with automatic window processing.
    ///
    /// For large batches (> 2× window size), uses parallel processing.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction fails.
    pub async fn add_dialogues(&mut self, dialogues: Vec<Dialogue>) -> Result<Vec<MemoryEntry>> {
        if dialogues.len() > self.window_size * 2 {
            return self.add_dialogues_parallel(dialogues).await;
        }

        let mut all_entries = Vec::new();
        self.dialogue_buffer.extend(dialogues);
        while self.dialogue_buffer.len() >= self.window_size {
            let entries = self.process_window().await?;
            all_entries.extend(entries);
        }
        Ok(all_entries)
    }

    /// Process remaining dialogues in the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction fails.
    pub async fn finalize(&mut self) -> Result<Vec<MemoryEntry>> {
        if self.dialogue_buffer.is_empty() {
            return Ok(Vec::new());
        }
        tracing::info!(
            remaining = self.dialogue_buffer.len(),
            "processing remaining dialogues"
        );
        let dialogues = std::mem::take(&mut self.dialogue_buffer);
        let entries = self.generate_entries(&dialogues).await?;
        self.processed_count += dialogues.len();
        entries.clone_into(&mut self.previous_entries);
        Ok(entries)
    }

    /// Returns the number of dialogues processed so far.
    #[must_use]
    pub const fn processed_count(&self) -> usize {
        self.processed_count
    }

    /// Returns the number of dialogues in the buffer.
    #[must_use]
    pub const fn buffer_len(&self) -> usize {
        self.dialogue_buffer.len()
    }

    async fn process_window(&mut self) -> Result<Vec<MemoryEntry>> {
        if self.dialogue_buffer.is_empty() {
            return Ok(Vec::new());
        }

        let end = self.window_size.min(self.dialogue_buffer.len());
        let window: Vec<Dialogue> = self.dialogue_buffer[..end].to_vec();
        let advance = self.step_size.min(self.dialogue_buffer.len());
        self.dialogue_buffer = self.dialogue_buffer[advance..].to_vec();

        tracing::info!(
            window_size = window.len(),
            processed = self.processed_count,
            "processing dialogue window"
        );

        let entries = self.generate_entries(&window).await?;
        self.processed_count += window.len();
        entries.clone_into(&mut self.previous_entries);

        tracing::info!(count = entries.len(), "generated memory entries");
        Ok(entries)
    }

    async fn add_dialogues_parallel(
        &mut self,
        dialogues: Vec<Dialogue>,
    ) -> Result<Vec<MemoryEntry>> {
        self.dialogue_buffer.extend(dialogues);
        let total_dialogues = self.dialogue_buffer.len();

        let mut windows = Vec::new();
        let mut pos = 0;
        while pos + self.window_size <= self.dialogue_buffer.len() {
            let window = self.dialogue_buffer[pos..pos + self.window_size].to_vec();
            windows.push(window);
            pos += self.step_size;
        }
        let remaining = self.dialogue_buffer[pos..].to_vec();
        if !remaining.is_empty() {
            windows.push(remaining);
        }
        self.dialogue_buffer.clear();

        tracing::info!(
            batches = windows.len(),
            workers = self.max_parallel_workers,
            "parallel processing dialogue windows"
        );

        let llm = Arc::clone(&self.llm);
        let context = Prompts::extraction_context(&self.previous_entries);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_parallel_workers));

        let mut handles = Vec::new();
        for (i, window) in windows.into_iter().enumerate() {
            let llm = Arc::clone(&llm);
            let ctx = context.clone();
            let sem = Arc::clone(&semaphore);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await;
                tracing::info!(window = i + 1, dialogues = window.len(), "worker started");
                let result = generate_entries_standalone(&llm, &window, &ctx).await;
                tracing::info!(window = i + 1, "worker finished");
                result
            }));
        }

        let mut all_entries = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(entries)) => all_entries.extend(entries),
                Ok(Err(e)) => tracing::error!(error = %e, "parallel window failed"),
                Err(e) => tracing::error!(error = %e, "task panicked"),
            }
        }

        self.processed_count += total_dialogues;
        if !all_entries.is_empty() {
            self.previous_entries = all_entries[all_entries.len().saturating_sub(10)..].to_vec();
        }

        Ok(all_entries)
    }

    async fn generate_entries(&self, dialogues: &[Dialogue]) -> Result<Vec<MemoryEntry>> {
        let context = Prompts::extraction_context(&self.previous_entries);
        generate_entries_standalone(&self.llm, dialogues, &context).await
    }
}

async fn generate_entries_standalone(
    llm: &Arc<dyn LlmClient>,
    dialogues: &[Dialogue],
    context: &str,
) -> Result<Vec<MemoryEntry>> {
    let dialogue_text: String = dialogues
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = Prompts::extraction(&dialogue_text, context);

    let messages = vec![
        Message::system(
            "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format.",
        ),
        Message::user(prompt),
    ];

    let opts = ChatOptions {
        temperature: 0.1,
        json_mode: false,
    };

    let parse_retries = 2;
    let mut last_err = None;
    for attempt in 0..=parse_retries {
        let response = match llm.chat(&messages, &opts).await {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        match parse_entries_response(&response) {
            Ok(entries) => return Ok(entries),
            Err(e) => {
                tracing::warn!(attempt = attempt + 1, error = %e, "parse failed, retrying LLM");
                last_err = Some(e);
            }
        }
    }

    Err(last_err
        .unwrap_or_else(|| Error::Internal("extraction parse retries exhausted".to_owned())))
}

fn parse_entries_response(response: &str) -> Result<Vec<MemoryEntry>> {
    let data = extract_json_from_text(response)?;
    let arr = data
        .as_array()
        .ok_or_else(|| Error::JsonParse("expected JSON array".to_owned()))?;

    let mut entries = Vec::with_capacity(arr.len());
    for item in arr {
        let restatement = item["lossless_restatement"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        if restatement.is_empty() {
            continue;
        }

        let keywords = item["keywords"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let timestamp = item["timestamp"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .and_then(|s| {
                DateTime::parse_from_rfc3339(s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
                    .or_else(|| {
                        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
                            .ok()
                            .map(|ndt| ndt.and_utc())
                    })
            });

        let location = item["location"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .map(String::from);

        let persons = item["persons"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let entities_list = item["entities"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let topic = item["topic"]
            .as_str()
            .filter(|s| *s != "null" && !s.is_empty())
            .map(String::from);

        entries.push(MemoryEntry {
            id: Uuid::new_v4(),
            restatement,
            keywords,
            timestamp,
            location,
            persons,
            entities: entities_list,
            topic,
        });
    }

    Ok(entries)
}
