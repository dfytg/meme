//! Configuration system with TOML file + environment variable support.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Top-level configuration for the meme system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// LLM provider configuration.
    pub llm: LlmConfig,
    /// Embedding model configuration.
    pub embedding: EmbeddingConfig,
    /// Storage configuration.
    pub store: StoreConfig,
    /// Pipeline parameters.
    pub pipeline: PipelineConfig,
    /// Parallel processing configuration.
    pub parallel: ParallelConfig,
    /// Cross-session configuration.
    pub cross: CrossConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            embedding: EmbeddingConfig::default(),
            store: StoreConfig::default(),
            pipeline: PipelineConfig::default(),
            parallel: ParallelConfig::default(),
            cross: CrossConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a TOML file, with environment variable overrides.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&content)?;
        config.apply_env_overrides();
        Ok(config)
    }

    /// Load configuration from the default location (`~/.meme/config.toml`),
    /// falling back to defaults if the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be parsed.
    pub fn load_default() -> Result<Self> {
        let path = default_config_path();
        if path.exists() {
            Self::from_file(&path)
        } else {
            let mut config = Self::default();
            config.apply_env_overrides();
            Ok(config)
        }
    }

    /// Save configuration to a TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("failed to serialize config: {e}")))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        Ok(())
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("MEME_LLM_API_KEY") {
            self.llm.api_key = Some(v);
        }
        if let Ok(v) = std::env::var("MEME_LLM_BASE_URL") {
            self.llm.base_url = v;
        }
        if let Ok(v) = std::env::var("MEME_LLM_MODEL") {
            self.llm.model = v;
        }
        if let Ok(v) = std::env::var("MEME_EMBEDDING_PROVIDER") {
            self.embedding.provider = v.parse().unwrap_or(self.embedding.provider);
        }
    }
}

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// API key (can also be set via `MEME_LLM_API_KEY`).
    pub api_key: Option<String>,
    /// Base URL for the OpenAI-compatible API.
    pub base_url: String,
    /// Model name.
    pub model: String,
    /// Enable streaming responses.
    pub streaming: bool,
    /// Temperature for generation.
    pub temperature: f32,
    /// Maximum retries for API calls.
    pub max_retries: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.openai.com/v1".to_owned(),
            model: "gpt-4.1-mini".to_owned(),
            streaming: true,
            temperature: 0.1,
            max_retries: 3,
        }
    }
}

/// Embedding provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderKind {
    /// Remote API-based embedding.
    #[default]
    Api,
    /// Local ONNX Runtime inference.
    Onnx,
}

impl std::str::FromStr for EmbeddingProviderKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "api" => Ok(Self::Api),
            "onnx" => Ok(Self::Onnx),
            other => Err(format!("unknown embedding provider: {other}")),
        }
    }
}

/// Embedding model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Which provider to use.
    pub provider: EmbeddingProviderKind,
    /// Model name (for API) or model identifier.
    pub model: String,
    /// Embedding dimension.
    pub dimension: usize,
    /// Path to ONNX model file (when provider = onnx).
    pub onnx_model_path: Option<String>,
    /// Path to tokenizer file (when provider = onnx).
    pub onnx_tokenizer_path: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProviderKind::Api,
            model: "text-embedding-3-small".to_owned(),
            dimension: 1024,
            onnx_model_path: None,
            onnx_tokenizer_path: None,
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StoreConfig {
    /// Path to LanceDB storage directory.
    pub lancedb_path: String,
    /// Memory table name.
    pub table_name: String,
}

impl Default for StoreConfig {
    fn default() -> Self {
        let base = default_data_dir();
        Self {
            lancedb_path: base.join("lancedb").to_string_lossy().into_owned(),
            table_name: "memories".to_owned(),
        }
    }
}

/// Pipeline parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    /// Number of dialogues per processing window.
    pub window_size: usize,
    /// Overlap between consecutive windows.
    pub overlap_size: usize,
    /// Max entries returned by semantic search.
    pub semantic_top_k: usize,
    /// Max entries returned by keyword search.
    pub keyword_top_k: usize,
    /// Max entries returned by structured search.
    pub structured_top_k: usize,
    /// Enable intent-aware retrieval planning.
    pub enable_planning: bool,
    /// Enable reflection-based additional retrieval.
    pub enable_reflection: bool,
    /// Maximum number of reflection rounds.
    pub max_reflection_rounds: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            window_size: 40,
            overlap_size: 2,
            semantic_top_k: 25,
            keyword_top_k: 5,
            structured_top_k: 5,
            enable_planning: true,
            enable_reflection: true,
            max_reflection_rounds: 2,
        }
    }
}

/// Parallel processing configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct ParallelConfig {
    /// Max concurrent workers for memory building.
    pub max_build_workers: usize,
    /// Max concurrent workers for retrieval queries.
    pub max_retrieval_workers: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_build_workers: 16,
            max_retrieval_workers: 8,
        }
    }
}

/// Cross-session configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CrossConfig {
    /// Path to the SQLite database for cross-session data.
    pub db_path: String,
    /// Maximum token budget for context injection.
    pub max_context_tokens: usize,
    /// Consolidation: entries older than this (days) receive decay.
    pub consolidation_max_age_days: u32,
    /// Consolidation: decay factor per period.
    pub consolidation_decay_factor: f64,
    /// Consolidation: cosine similarity threshold for merging.
    pub consolidation_merge_threshold: f64,
    /// Consolidation: minimum importance before pruning.
    pub consolidation_min_importance: f64,
    /// Consolidation: max entries processed per run.
    pub consolidation_max_entries_per_run: usize,
}

impl Default for CrossConfig {
    fn default() -> Self {
        let base = default_data_dir();
        Self {
            db_path: base.join("cross.db").to_string_lossy().into_owned(),
            max_context_tokens: 4096,
            consolidation_max_age_days: 90,
            consolidation_decay_factor: 0.9,
            consolidation_merge_threshold: 0.95,
            consolidation_min_importance: 0.05,
            consolidation_max_entries_per_run: 1000,
        }
    }
}

/// Returns the default data directory (`~/.meme/`).
fn default_data_dir() -> PathBuf {
    directories::BaseDirs::new()
        .map(|d| d.home_dir().join(".meme"))
        .unwrap_or_else(|| PathBuf::from(".meme"))
}

/// Returns the default config file path (`~/.meme/config.toml`).
pub fn default_config_path() -> PathBuf {
    default_data_dir().join("config.toml")
}
