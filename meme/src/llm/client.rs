//! OpenAI-compatible async LLM client.

use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;
use crate::error::{Error, Result};

/// Chat message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt.
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: Role,
    /// Message content.
    pub content: String,
}

impl Message {
    /// Create a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// Create a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Create an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Options for a chat completion request.
#[derive(Debug, Clone, Copy)]
pub struct ChatOptions {
    /// Temperature for generation.
    pub temperature: f32,
    /// Whether to request JSON output format.
    pub json_mode: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            json_mode: false,
        }
    }
}

/// OpenAI-compatible HTTP LLM client.
#[derive(Debug, Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_retries: u32,
}

impl LlmClient {
    /// Create a new client from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is missing.
    pub fn from_config(config: &LlmConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| Error::Config("LLM API key is required".to_owned()))?;

        Ok(Self {
            http: reqwest::Client::new(),
            base_url: config.base_url.trim_end_matches('/').to_owned(),
            api_key,
            model: config.model.clone(),
            max_retries: config.max_retries,
        })
    }

    /// Create a new client with explicit parameters.
    #[must_use]
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            max_retries: 3,
        }
    }

    /// Send a chat completion request and return the response text.
    ///
    /// Retries with exponential backoff on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if the API call fails after retries.
    pub async fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut last_err = None;
        for attempt in 0..self.max_retries {
            match self.call_api(messages, opts).await {
                Ok(content) => return Ok(content),
                Err(e) => {
                    tracing::warn!(attempt = attempt + 1, error = %e, "LLM API call failed");
                    last_err = Some(e);
                    if attempt + 1 < self.max_retries {
                        let wait = 1u64 << attempt;
                        tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Llm("all retries exhausted".to_owned())))
    }

    async fn call_api(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": opts.temperature,
            "stream": false,
        });

        if opts.json_mode {
            body["response_format"] = serde_json::json!({"type": "json_object"});
        }

        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Llm(format!("API returned {status}: {text}")));
        }

        let data: serde_json::Value = resp.json().await?;
        data["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| Error::Llm("missing content in API response".to_owned()))
    }
}

/// Extract a JSON value from text that may contain markdown fences and other noise.
///
/// # Errors
///
/// Returns an error if no valid JSON can be found in the input.
pub fn extract_json_from_text(text: &str) -> Result<serde_json::Value> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::JsonParse("empty response".to_owned()));
    }

    // Strip common LLM prefixes.
    let stripped = strip_prefixes(text);

    // Try direct parse.
    if let Ok(v) = serde_json::from_str(stripped) {
        return Ok(v);
    }

    // Try extracting from ```json ... ``` block.
    if let Some(json_str) = extract_fenced_json(stripped)
        && let Ok(v) = parse_with_cleanup(&json_str)
    {
        return Ok(v);
    }

    // Try extracting from generic ``` ... ``` block.
    if let Some(json_str) = extract_generic_fenced(stripped)
        && let Ok(v) = parse_with_cleanup(&json_str)
    {
        return Ok(v);
    }

    // Try finding balanced JSON object/array.
    for start_char in ['{', '['] {
        if let Some(v) = extract_balanced_json(stripped, start_char) {
            return Ok(v);
        }
    }

    Err(Error::JsonParse(format!(
        "no valid JSON found in: {}...",
        &text[..text.len().min(200)]
    )))
}

fn strip_prefixes(text: &str) -> &str {
    let prefixes = [
        "here's the json:",
        "here is the json:",
        "the json is:",
        "json:",
        "result:",
        "output:",
        "answer:",
    ];
    let lower = text.to_lowercase();
    for prefix in prefixes {
        if lower.starts_with(prefix) {
            return text[prefix.len()..].trim();
        }
    }
    text
}

fn extract_fenced_json(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let start_marker = "```json";
    let start_idx = lower.find(start_marker)?;
    let content_start = start_idx + start_marker.len();
    let end_idx = text[content_start..].find("```")?;
    Some(
        text[content_start..content_start + end_idx]
            .trim()
            .to_owned(),
    )
}

fn extract_generic_fenced(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after_fence = start + 3;
    // Skip language identifier on the same line.
    let newline = text[after_fence..].find('\n')?;
    let content_start = after_fence + newline + 1;
    let end = text[content_start..].find("```")?;
    Some(text[content_start..content_start + end].trim().to_owned())
}

fn parse_with_cleanup(json_str: &str) -> std::result::Result<serde_json::Value, ()> {
    if let Ok(v) = serde_json::from_str(json_str) {
        return Ok(v);
    }
    let cleaned = cleanup_json(json_str);
    serde_json::from_str(&cleaned).map_err(|_| ())
}

fn cleanup_json(s: &str) -> String {
    use std::sync::LazyLock;

    static RE_TRAILING_COMMA: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r",(\s*[}\]])").expect("valid regex"));
    static RE_LINE_COMMENT: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"(?m)//.*$").expect("valid regex"));

    let s = RE_TRAILING_COMMA.replace_all(s, "$1");
    RE_LINE_COMMENT.replace_all(&s, "").trim().to_owned()
}

fn extract_balanced_json(text: &str, start_char: char) -> Option<serde_json::Value> {
    let end_char = if start_char == '{' { '}' } else { ']' };
    let start_idx = text.find(start_char)?;

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in text[start_idx..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == start_char {
            depth += 1;
        } else if ch == end_char {
            depth -= 1;
            if depth == 0 {
                let json_str = &text[start_idx..=(start_idx + i)];
                if let Ok(v) = serde_json::from_str(json_str) {
                    return Some(v);
                }
                let cleaned = cleanup_json(json_str);
                if let Ok(v) = serde_json::from_str(&cleaned) {
                    return Some(v);
                }
                break;
            }
        }
    }
    None
}
