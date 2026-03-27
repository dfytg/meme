//! OpenAI-compatible async LLM client.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;
use crate::error::{Error, Result};
use crate::llm::json::extract_json_from_text;

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
            json_mode: true,
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
    /// Create a new client from configuration using a shared HTTP client.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is missing.
    pub fn new(http: reqwest::Client, config: &LlmConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| Error::Config("LLM API key is required".to_owned()))?;

        Ok(Self {
            http,
            base_url: config.base_url.trim_end_matches('/').to_owned(),
            api_key,
            model: config.model.clone(),
            max_retries: config.max_retries,
        })
    }

    /// Send a chat completion and deserialize the response into `T`.
    ///
    /// Uses `json_object` response format + `serde_json::from_str` for type-safe parsing.
    /// Retries with exponential backoff on transient or parse failures.
    ///
    /// # Errors
    ///
    /// Returns an error if the API call fails after retries or the response
    /// cannot be deserialized into `T`.
    #[tracing::instrument(skip(self, messages, opts), fields(model = %self.model))]
    pub async fn chat_structured<T: serde::de::DeserializeOwned>(
        &self,
        messages: &[Message],
        opts: &ChatOptions,
    ) -> Result<T> {
        let mut last_err = None;
        for attempt in 0..self.max_retries {
            match self.call_api(messages, opts).await {
                Ok(content) => match serde_json::from_str::<T>(&content) {
                    Ok(parsed) => return Ok(parsed),
                    Err(e) => {
                        tracing::warn!(attempt = attempt + 1, error = %e, "JSON parse failed");
                        if let Ok(v) = extract_json_from_text(&content)
                            && let Ok(parsed) = serde_json::from_value::<T>(v)
                        {
                            return Ok(parsed);
                        }
                        last_err = Some(Error::JsonParse(format!("{e}")));
                    }
                },
                Err(e) => {
                    if !e.is_retryable() {
                        return Err(e);
                    }
                    tracing::warn!(attempt = attempt + 1, error = %e, "LLM API call failed");
                    last_err = Some(e);
                }
            }
            if attempt + 1 < self.max_retries {
                let wait = 2u64.saturating_pow(attempt).min(30);
                tokio::time::sleep(Duration::from_secs(wait)).await;
            }
        }
        Err(last_err.unwrap_or_else(|| Error::llm("all retries exhausted")))
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
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::llm_with_status(
                status.as_u16(),
                format!("API returned {status}: {text}"),
            ));
        }

        let data: serde_json::Value = resp.json().await?;
        data["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| Error::llm("missing content in API response"))
    }
}

