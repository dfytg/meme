//! Observation extractor — derives structured observations from session events.
//!
//! Converts raw session events into `CrossObservation` objects and `Dialogue` objects
//! using rule-based heuristics. Also estimates session value/importance.

use uuid::Uuid;

use crate::model::{CrossObservation, Dialogue, EventKind, ObservationType, SessionEvent};

/// Extracts structured observations from collected session events.
#[derive(Debug, Clone, Copy)]
pub struct ObservationExtractor;

impl ObservationExtractor {
    /// Extract `CrossObservation` objects from a list of session events.
    ///
    /// Maps event kinds to observation types:
    /// - `ToolUse` / `FileChange` → `Change`
    /// - `Message` / `Note` → `Discovery`
    #[must_use]
    pub fn extract_from_events(
        events: &[SessionEvent],
        memory_session_id: &Uuid,
    ) -> Vec<CrossObservation> {
        events
            .iter()
            .filter_map(|event| {
                let title = event.title.as_deref()?;
                if title.is_empty() {
                    return None;
                }

                let obs_type = match event.kind {
                    EventKind::ToolUse | EventKind::FileChange => ObservationType::Change,
                    EventKind::Message | EventKind::Note | EventKind::System => {
                        ObservationType::Discovery
                    }
                };

                let narrative = event.payload_json.as_deref().and_then(|json_str| {
                    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
                    let text = v
                        .get("content")
                        .or_else(|| v.get("tool_output"))
                        .or_else(|| v.get("note"))
                        .and_then(|v| v.as_str())?;
                    if text.is_empty() {
                        return None;
                    }
                    if text.len() > 500 {
                        let boundary = text.floor_char_boundary(497);
                        Some(format!("{}...", &text[..boundary]))
                    } else {
                        Some(text.to_owned())
                    }
                });

                Some(CrossObservation {
                    obs_id: None,
                    memory_session_id: *memory_session_id,
                    timestamp: event.timestamp,
                    obs_type,
                    title: title.to_owned(),
                    narrative,
                })
            })
            .collect()
    }

    /// Convert session events to `Dialogue` objects for the core `MemoryBuilder` pipeline.
    ///
    /// - Messages become `role: content` dialogues
    /// - Tool uses become `"Agent: Used tool X. Input: ... Output: ..."` dialogues
    /// - File changes become `"System: File X was modified"` dialogues
    /// - Notes become `"System: note content"` dialogues
    #[must_use]
    pub fn events_to_dialogues(events: &[SessionEvent]) -> Vec<Dialogue> {
        events
            .iter()
            .filter_map(|event| {
                let payload: serde_json::Value = event
                    .payload_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Null);

                let (speaker, content) = match event.kind {
                    EventKind::Message => {
                        let role = payload
                            .get("role")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Speaker");
                        let content = payload
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim();
                        if content.is_empty() {
                            return None;
                        }
                        (role.to_owned(), content.to_owned())
                    }
                    EventKind::ToolUse => {
                        let tool_name = payload
                            .get("tool_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("tool");
                        let tool_input = payload
                            .get("tool_input")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let tool_output = payload
                            .get("tool_output")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let mut parts = vec![format!("Used tool {tool_name}.")];
                        if !tool_input.is_empty() {
                            parts.push(format!("Input: {tool_input}"));
                        }
                        if !tool_output.is_empty() {
                            parts.push(format!("Output: {tool_output}"));
                        }
                        ("Agent".to_owned(), parts.join(" "))
                    }
                    EventKind::FileChange => {
                        let filepath = payload
                            .get("filepath")
                            .and_then(|v| v.as_str())
                            .unwrap_or("file");
                        let change_type = payload
                            .get("change_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("modified");
                        (
                            "System".to_owned(),
                            format!("File {filepath} was {change_type}."),
                        )
                    }
                    EventKind::Note => {
                        let note = payload
                            .get("note")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim();
                        if note.is_empty() {
                            return None;
                        }
                        ("System".to_owned(), note.to_owned())
                    }
                    EventKind::System => return None,
                };

                let mut dialogue = Dialogue::new(speaker, content);
                dialogue = dialogue.with_timestamp(event.timestamp);
                Some(dialogue)
            })
            .collect()
    }

    /// Estimate the value/importance of a session based on its events.
    ///
    /// Higher value for sessions with more tool usage, file modifications,
    /// longer conversations, and diverse event types.
    ///
    /// Returns a float between 0.0 and 1.0.
    #[must_use]
    pub fn estimate_session_value(events: &[SessionEvent]) -> f64 {
        if events.is_empty() {
            return 0.0;
        }

        let message_count = events
            .iter()
            .filter(|e| e.kind == EventKind::Message)
            .count();
        let tool_count = events
            .iter()
            .filter(|e| e.kind == EventKind::ToolUse)
            .count();
        let file_change_count = events
            .iter()
            .filter(|e| e.kind == EventKind::FileChange)
            .count();
        let note_count = events.iter().filter(|e| e.kind == EventKind::Note).count();

        let modified_file_hits: usize = events
            .iter()
            .filter(|e| e.kind == EventKind::ToolUse)
            .filter_map(|e| e.payload_json.as_deref())
            .filter_map(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .filter_map(|v| v.get("files_modified")?.as_array().map(Vec::len))
            .sum();

        let mut value = 0.1;
        value += (message_count as f64 * 0.02).min(0.3);
        value += (tool_count as f64 * 0.08).min(0.5);
        value += (file_change_count as f64 * 0.12).min(0.5);
        value += (modified_file_hits as f64 * 0.05).min(0.4);

        let diversity = [
            message_count > 0,
            tool_count > 0,
            file_change_count > 0,
            note_count > 0,
        ]
        .iter()
        .filter(|&&v| v)
        .count();
        if diversity >= 3 {
            value += 0.1;
        }

        value.clamp(0.0, 1.0)
    }
}
