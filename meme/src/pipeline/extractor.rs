//! Stage 1: Semantic Structured Compression.
//!
//! The [`Extractor`] buffers incoming dialogues, splits them into overlapping
//! windows, and sends each window to an LLM that produces atomic, self-contained
//! [`Memory`](crate::model::Memory) entries.  Large batches are processed in
//! parallel via tokio tasks bounded by a semaphore.

use std::sync::Arc;

use crate::config::PipelineConfig;
use crate::error::{Error, Result};
use crate::llm::{ChatOptions, ExtractionResponse, LlmClient, Message, prompt};
use crate::model::{Dialogue, Memory};

/// Dialogue-to-memory extractor implementing Stage 1 (Semantic Structured Compression)
/// of the pipeline.
pub struct Extractor {
    llm: Arc<LlmClient>,
    window_size: usize,
    overlap_size: usize,
    step_size: usize,
    max_parallel_workers: usize,
    custom_extraction_prompt: Option<String>,
    dialogue_buffer: Vec<Dialogue>,
    processed_count: usize,
    previous_entries: Vec<Memory>,
}

impl std::fmt::Debug for Extractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extractor")
            .field("window_size", &self.window_size)
            .field("overlap_size", &self.overlap_size)
            .field("buffer_len", &self.dialogue_buffer.len())
            .field("processed_count", &self.processed_count)
            .finish()
    }
}

impl Extractor {
    /// Create a new extractor.
    #[must_use]
    pub fn new(
        llm: Arc<LlmClient>,
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
            custom_extraction_prompt: pipeline_cfg.custom_extraction_prompt.clone(),
            dialogue_buffer: Vec::new(),
            processed_count: 0,
            previous_entries: Vec::new(),
        }
    }

    /// Add dialogues to the buffer with automatic window processing.
    ///
    /// For large batches (> 2× window size), uses parallel processing.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction fails.
    #[tracing::instrument(skip(self, dialogues), fields(count = dialogues.len()))]
    pub async fn add_dialogues(&mut self, dialogues: Vec<Dialogue>) -> Result<Vec<Memory>> {
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

    /// Flush the dialogue buffer — process any remaining buffered dialogues.
    ///
    /// # Errors
    ///
    /// Returns an error if LLM extraction fails.
    #[tracing::instrument(skip(self), fields(remaining = self.dialogue_buffer.len()))]
    pub async fn flush(&mut self) -> Result<Vec<Memory>> {
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

    async fn process_window(&mut self) -> Result<Vec<Memory>> {
        if self.dialogue_buffer.is_empty() {
            return Ok(Vec::new());
        }

        let end = self.window_size.min(self.dialogue_buffer.len());
        let window: Vec<Dialogue> = self.dialogue_buffer[..end].to_vec();
        let advance = self.step_size.min(self.dialogue_buffer.len());
        drop(self.dialogue_buffer.drain(..advance));

        tracing::info!(
            window_size = window.len(),
            processed = self.processed_count,
            "processing dialogue window"
        );

        let entries = self.generate_entries(&window).await?;
        self.processed_count += advance;
        entries.clone_into(&mut self.previous_entries);

        tracing::info!(count = entries.len(), "generated memory entries");
        Ok(entries)
    }

    async fn add_dialogues_parallel(&mut self, dialogues: Vec<Dialogue>) -> Result<Vec<Memory>> {
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
        let context = prompt::extraction_context(&self.previous_entries);
        let custom_prompt = self.custom_extraction_prompt.clone();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_parallel_workers));

        let mut handles = Vec::new();
        for (i, window) in windows.into_iter().enumerate() {
            let llm = Arc::clone(&llm);
            let ctx = context.clone();
            let cp = custom_prompt.clone();
            let sem = Arc::clone(&semaphore);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await;
                tracing::info!(window = i + 1, dialogues = window.len(), "worker started");
                let result = generate_entries_standalone(&llm, &window, &ctx, cp.as_deref()).await;
                tracing::info!(window = i + 1, "worker finished");
                result
            }));
        }

        let mut all_entries = Vec::new();
        let mut errors = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(entries)) => all_entries.extend(entries),
                Ok(Err(e)) => {
                    tracing::error!(error = %e, "parallel window failed");
                    errors.push(e);
                }
                Err(e) => {
                    tracing::error!(error = %e, "task panicked");
                    errors.push(Error::Internal(format!("task panicked: {e}")));
                }
            }
        }
        if !errors.is_empty() && all_entries.is_empty() {
            return Err(errors.into_iter().next().expect("non-empty errors"));
        }
        if !errors.is_empty() {
            tracing::warn!(
                failed = errors.len(),
                succeeded = all_entries.len(),
                "partial failure in parallel processing"
            );
        }

        self.processed_count += total_dialogues;
        if !all_entries.is_empty() {
            self.previous_entries = all_entries[all_entries.len().saturating_sub(10)..].to_vec();
        }

        Ok(all_entries)
    }

    async fn generate_entries(&self, dialogues: &[Dialogue]) -> Result<Vec<Memory>> {
        let context = prompt::extraction_context(&self.previous_entries);
        generate_entries_standalone(
            &self.llm,
            dialogues,
            &context,
            self.custom_extraction_prompt.as_deref(),
        )
        .await
    }
}

async fn generate_entries_standalone(
    llm: &Arc<LlmClient>,
    dialogues: &[Dialogue],
    context: &str,
    custom_prompt: Option<&str>,
) -> Result<Vec<Memory>> {
    let dialogue_text: String = dialogues
        .iter()
        .map(Dialogue::format_for_prompt)
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = custom_prompt.map_or_else(
        || prompt::extraction(&dialogue_text, context),
        |cp| format!("{cp}\n\n[Dialogues]\n{dialogue_text}\n\n{context}"),
    );

    let messages = vec![
        Message::system(
            "You are a professional information extraction assistant. You must output valid JSON.",
        ),
        Message::user(prompt),
    ];
    let opts = ChatOptions {
        temperature: 0.1,
        json_mode: true,
    };

    let response: ExtractionResponse = llm.chat_structured(&messages, &opts).await?;
    Ok(response
        .entries
        .into_iter()
        .filter_map(crate::llm::ExtractedEntry::into_memory)
        .collect())
}
