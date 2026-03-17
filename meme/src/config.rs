//! Configuration system with TOML file + environment variable support.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Top-level configuration for the meme system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    /// LLM provider configuration.
    pub llm: LlmConfig,
    /// Embedding model configuration.
    pub embedding: EmbeddingConfig,
    /// Storage configuration.
    pub store: StoreConfig,
    /// Pipeline parameters.
    pub pipeline: PipelineConfig,
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

    /// Validate configuration for common mistakes.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration value is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.pipeline.overlap_size >= self.pipeline.window_size {
            return Err(Error::Config(format!(
                "overlap_size ({}) must be less than window_size ({})",
                self.pipeline.overlap_size, self.pipeline.window_size
            )));
        }
        if self.pipeline.window_size == 0 {
            return Err(Error::Config("window_size must be > 0".into()));
        }
        if self.embedding.dimension == 0 {
            return Err(Error::Config("embedding dimension must be > 0".into()));
        }
        if self.llm.max_retries == 0 {
            return Err(Error::Config("max_retries must be > 0".into()));
        }
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
    /// Model name — API model name (e.g. `"text-embedding-3-small"`) or
    /// fastembed model code (e.g. `"BAAI/bge-small-en-v1.5"`).
    pub model: String,
    /// Embedding dimension (used by API provider; auto-detected for ONNX).
    pub dimension: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProviderKind::Api,
            model: "text-embedding-3-small".to_owned(),
            dimension: 1536,
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StoreConfig {
    /// Path to `LanceDB` storage directory.
    pub lancedb_path: String,
    /// Path to the `SQLite` history database file.
    pub history_db_path: String,
    /// Memory table name.
    pub table_name: String,
}

impl Default for StoreConfig {
    fn default() -> Self {
        let base = default_data_dir();
        Self {
            lancedb_path: base.join("lancedb").to_string_lossy().into_owned(),
            history_db_path: base.join("history.db").to_string_lossy().into_owned(),
            table_name: "memories".to_owned(),
        }
    }
}

/// Pipeline parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Max concurrent workers for memory building.
    pub max_build_workers: usize,
    /// Max concurrent workers for retrieval queries.
    pub max_retrieval_workers: usize,
    /// Custom extraction prompt (replaces the built-in extraction prompt).
    pub custom_extraction_prompt: Option<String>,
    /// Custom answer generation prompt (replaces the built-in answer prompt).
    pub custom_answer_prompt: Option<String>,
    /// Enable LLM-based reranking of search results.
    pub enable_rerank: bool,
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
            max_build_workers: 16,
            max_retrieval_workers: 8,
            custom_extraction_prompt: None,
            custom_answer_prompt: None,
            enable_rerank: false,
        }
    }
}

/// Returns the default data directory (`~/.meme/`).
fn default_data_dir() -> PathBuf {
    directories::BaseDirs::new()
        .map_or_else(|| PathBuf::from(".meme"), |d| d.home_dir().join(".meme"))
}

/// Returns the default config file path (`~/.meme/config.toml`).
#[must_use]
pub fn default_config_path() -> PathBuf {
    default_data_dir().join("config.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_values() {
        let c = Config::default();
        assert!(c.llm.api_key.is_none());
        assert_eq!(c.llm.model, "gpt-4.1-mini");
        assert_eq!(c.llm.max_retries, 3);
        assert!((c.llm.temperature - 0.1).abs() < f32::EPSILON);
        assert_eq!(c.embedding.provider, EmbeddingProviderKind::Api);
        assert_eq!(c.embedding.dimension, 1536);
        assert_eq!(c.pipeline.window_size, 40);
        assert!(c.pipeline.enable_planning);
        assert!(c.pipeline.enable_reflection);
    }

    #[test]
    fn config_toml_roundtrip() {
        let c = Config::default();
        let toml_str = toml::to_string_pretty(&c).unwrap();
        let c2: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(c2.llm.model, c.llm.model);
        assert_eq!(c2.embedding.dimension, c.embedding.dimension);
        assert_eq!(c2.pipeline.window_size, c.pipeline.window_size);
    }

    #[test]
    fn embedding_provider_from_str() {
        assert_eq!(
            "api".parse::<EmbeddingProviderKind>().unwrap(),
            EmbeddingProviderKind::Api
        );
        assert_eq!(
            "API".parse::<EmbeddingProviderKind>().unwrap(),
            EmbeddingProviderKind::Api
        );
        assert_eq!(
            "onnx".parse::<EmbeddingProviderKind>().unwrap(),
            EmbeddingProviderKind::Onnx
        );
        assert_eq!(
            "ONNX".parse::<EmbeddingProviderKind>().unwrap(),
            EmbeddingProviderKind::Onnx
        );
        assert!("unknown".parse::<EmbeddingProviderKind>().is_err());
    }

    #[test]
    fn pipeline_config_overlap_guard() {
        let c = PipelineConfig::default();
        let step = c.window_size.saturating_sub(c.overlap_size).max(1);
        assert!(step > 0);
        assert!(step <= c.window_size);
    }

    #[test]
    fn validate_default_ok() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn validate_overlap_ge_window() {
        let mut c = Config::default();
        c.pipeline.overlap_size = c.pipeline.window_size;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_zero_window() {
        let mut c = Config::default();
        c.pipeline.window_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_zero_dimension() {
        let mut c = Config::default();
        c.embedding.dimension = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_zero_retries() {
        let mut c = Config::default();
        c.llm.max_retries = 0;
        assert!(c.validate().is_err());
    }
}
