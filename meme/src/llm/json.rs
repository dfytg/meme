//! JSON extraction fallback — defensive parsing for non-standard LLM output.
//!
//! With `json_object` response format, OpenAI-compatible APIs guarantee valid JSON.
//! This module only activates as a fallback when direct `serde_json::from_str` fails
//! (e.g. non-OpenAI providers wrapping JSON in markdown fences).

use crate::error::{MemeError, Result};

/// Extract a JSON value from text that may be wrapped in markdown fences.
///
/// Fallback path only — callers should try `serde_json::from_str` first.
///
/// # Errors
///
/// Returns an error if no valid JSON can be found.
pub(super) fn extract_json_from_text(text: &str) -> Result<serde_json::Value> {
    let text = text.trim();
    if text.is_empty() {
        return Err(MemeError::JsonParse("empty response".to_owned()));
    }

    if let Ok(v) = serde_json::from_str(text) {
        return Ok(v);
    }

    if let Some(inner) = extract_fenced(text)
        && let Ok(v) = serde_json::from_str(&inner)
    {
        return Ok(v);
    }

    if let Some(v) = extract_balanced(text, '{', '}') {
        return Ok(v);
    }
    if let Some(v) = extract_balanced(text, '[', ']') {
        return Ok(v);
    }

    Err(MemeError::JsonParse(format!(
        "no valid JSON found in: {}...",
        &text[..text.len().min(200)]
    )))
}

/// Extract content from a ``` or ```json fenced block.
fn extract_fenced(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after_fence = start + 3;
    let newline = text[after_fence..].find('\n')?;
    let content_start = after_fence + newline + 1;
    let end = text[content_start..].find("```")?;
    Some(text[content_start..content_start + end].trim().to_owned())
}

/// Find the first balanced `{…}` or `[…]` and try to parse it.
fn extract_balanced(text: &str, open: char, close: char) -> Option<serde_json::Value> {
    let start = text.find(open)?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in text[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' => escape = true,
            '"' => in_string = !in_string,
            c if !in_string && c == open => depth += 1,
            c if !in_string && c == close => {
                depth -= 1;
                if depth == 0 {
                    let slice = &text[start..=(start + i)];
                    return serde_json::from_str(slice).ok();
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(extract_json_from_text("").is_err());
        assert!(extract_json_from_text("   ").is_err());
    }

    #[test]
    fn direct_json() {
        let obj = extract_json_from_text(r#"{"key": "value"}"#).unwrap();
        assert_eq!(obj.get("key").and_then(|v| v.as_str()), Some("value"));

        let arr = extract_json_from_text(r"[1, 2, 3]").unwrap();
        assert_eq!(arr.as_array().unwrap().len(), 3);
    }

    #[test]
    fn fenced_block() {
        let tagged = "Here:\n```json\n{\"a\": 1}\n```\nDone.";
        assert_eq!(
            extract_json_from_text(tagged)
                .unwrap()
                .get("a")
                .and_then(serde_json::Value::as_i64),
            Some(1)
        );

        let untagged = "Result:\n```\n{\"b\": 2}\n```";
        assert_eq!(
            extract_json_from_text(untagged)
                .unwrap()
                .get("b")
                .and_then(serde_json::Value::as_i64),
            Some(2)
        );
    }

    #[test]
    fn balanced_in_text() {
        let input = r#"The answer is {"name": "Alice", "age": 30} done."#;
        let v = extract_json_from_text(input).unwrap();
        assert_eq!(v.get("name").and_then(|v| v.as_str()), Some("Alice"));
        assert_eq!(v.get("age").and_then(serde_json::Value::as_i64), Some(30));
    }

    #[test]
    fn nested_objects() {
        let v = extract_json_from_text(r#"{"outer": {"inner": [1, 2]}}"#).unwrap();
        let inner = v
            .get("outer")
            .and_then(|o| o.get("inner"))
            .and_then(|a| a.as_array());
        assert_eq!(inner.map(Vec::len), Some(2));
    }

    #[test]
    fn no_valid_json() {
        assert!(extract_json_from_text("just plain text").is_err());
        assert!(extract_json_from_text("not {valid json").is_err());
    }
}
