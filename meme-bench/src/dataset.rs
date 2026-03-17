//! LOCOMO-compatible benchmark dataset format.

use serde::{Deserialize, Serialize};

/// A complete benchmark dataset containing conversations and QA pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    /// Dataset name / identifier.
    pub name: String,
    /// Individual conversation scenarios to evaluate.
    pub scenarios: Vec<Scenario>,
}

/// A single conversation scenario with dialogues and evaluation questions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Scenario identifier.
    pub id: String,
    /// Description of the conversation context.
    #[serde(default)]
    pub description: String,
    /// Ordered dialogue turns to feed into the memory system.
    pub dialogues: Vec<DialogueTurn>,
    /// Questions to evaluate after all dialogues are ingested.
    pub questions: Vec<Question>,
}

/// A single dialogue turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    /// Speaker name.
    pub speaker: String,
    /// Dialogue content.
    pub content: String,
    /// Optional ISO 8601 timestamp.
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// LOCOMO question categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionCategory {
    /// Requires recalling a single fact from the conversation.
    SingleHop,
    /// Requires combining multiple facts from different parts.
    MultiHop,
    /// Requires understanding temporal relationships and ordering.
    Temporal,
    /// Questions that require commonsense or world knowledge beyond the conversation.
    Commonsense,
    /// Adversarial questions designed to test hallucination resistance.
    Adversarial,
}

impl std::fmt::Display for QuestionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleHop => write!(f, "single_hop"),
            Self::MultiHop => write!(f, "multi_hop"),
            Self::Temporal => write!(f, "temporal"),
            Self::Commonsense => write!(f, "commonsense"),
            Self::Adversarial => write!(f, "adversarial"),
        }
    }
}

/// A benchmark evaluation question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// Question identifier.
    pub id: String,
    /// The question text.
    pub question: String,
    /// Ground truth answer.
    pub answer: String,
    /// Question category for per-type scoring.
    pub category: QuestionCategory,
    /// Optional list of acceptable alternative answers.
    #[serde(default)]
    pub acceptable_answers: Vec<String>,
}

/// Raw LOCOMO JSON format (locomo10.json).
mod raw {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct LocomoEntry {
        pub qa: Vec<RawQuestion>,
        pub conversation: Conversation,
    }

    #[derive(Debug, Deserialize)]
    pub struct RawQuestion {
        pub question: String,
        pub answer: Option<serde_json::Value>,
        #[serde(default)]
        #[allow(dead_code)]
        pub evidence: Vec<String>,
        pub category: u8,
        #[allow(dead_code)]
        pub adversarial_answer: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    pub struct Conversation {
        pub speaker_a: String,
        pub speaker_b: String,
        #[serde(flatten)]
        pub sessions: std::collections::BTreeMap<String, serde_json::Value>,
    }

    #[derive(Debug, Deserialize)]
    pub struct DialogueItem {
        pub speaker: String,
        pub text: String,
        #[serde(default)]
        #[allow(dead_code)]
        pub dia_id: Option<String>,
    }
}

/// Load a LOCOMO-format dataset (locomo10.json) and convert to [`BenchmarkDataset`].
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_locomo(path: &std::path::Path) -> Result<BenchmarkDataset, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let entries: Vec<raw::LocomoEntry> =
        serde_json::from_str(&content).map_err(|e| format!("failed to parse LOCOMO JSON: {e}"))?;

    let mut scenarios = Vec::with_capacity(entries.len());
    for (idx, entry) in entries.into_iter().enumerate() {
        let dialogues = extract_dialogues(&entry.conversation);
        let questions = entry
            .qa
            .into_iter()
            .enumerate()
            .map(|(qi, q)| convert_question(idx, qi, q))
            .collect();

        scenarios.push(Scenario {
            id: format!("locomo_{idx}"),
            description: format!(
                "{} & {} conversation",
                entry.conversation.speaker_a, entry.conversation.speaker_b
            ),
            dialogues,
            questions,
        });
    }

    Ok(BenchmarkDataset {
        name: "LOCOMO".into(),
        scenarios,
    })
}

fn extract_dialogues(conv: &raw::Conversation) -> Vec<DialogueTurn> {
    let mut dialogues = Vec::new();

    let mut session_keys: Vec<&String> = conv
        .sessions
        .keys()
        .filter(|k| k.starts_with("session_") && !k.ends_with("_date_time"))
        .collect();
    session_keys.sort_by(|a, b| {
        let num_a = a.trim_start_matches("session_").parse::<u32>().unwrap_or(0);
        let num_b = b.trim_start_matches("session_").parse::<u32>().unwrap_or(0);
        num_a.cmp(&num_b)
    });

    for key in session_keys {
        let date_key = format!("{key}_date_time");
        let session_ts = conv.sessions.get(&date_key).and_then(|v| v.as_str()).map(String::from);

        let Some(session_val) = conv.sessions.get(key) else {
            continue;
        };
        let Ok(items) = serde_json::from_value::<Vec<raw::DialogueItem>>(session_val.clone())
        else {
            continue;
        };

        for item in items {
            dialogues.push(DialogueTurn {
                speaker: item.speaker,
                content: item.text,
                timestamp: session_ts.clone(),
            });
        }
    }
    dialogues
}

fn map_category(cat: u8) -> QuestionCategory {
    match cat {
        1 => QuestionCategory::SingleHop,
        2 => QuestionCategory::Temporal,
        3 => QuestionCategory::Commonsense,
        4 => QuestionCategory::MultiHop,
        5 => QuestionCategory::Adversarial,
        _ => QuestionCategory::SingleHop,
    }
}

fn convert_question(scenario_idx: usize, q_idx: usize, q: raw::RawQuestion) -> Question {
    let category = map_category(q.category);

    let answer = if category == QuestionCategory::Adversarial {
        // Adversarial questions have no ground-truth answer.
        // The correct response is to indicate the information doesn't apply.
        "unknown".to_owned()
    } else {
        q.answer
            .map(|v| match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            })
            .unwrap_or_default()
    };

    let mut acceptable = Vec::new();
    if category == QuestionCategory::Adversarial {
        // For adversarial: accepting refusal-style answers
        acceptable.extend([
            "I don't know".to_owned(),
            "not mentioned".to_owned(),
            "no information".to_owned(),
            "cannot determine".to_owned(),
            "not available".to_owned(),
        ]);
    }

    Question {
        id: format!("s{scenario_idx}_q{q_idx}"),
        question: q.question,
        answer,
        category,
        acceptable_answers: acceptable,
    }
}

/// Generate a sample benchmark dataset for testing/demonstration.
#[must_use]
pub fn sample_dataset() -> BenchmarkDataset {
    BenchmarkDataset {
        name: "meme-sample-bench".into(),
        scenarios: vec![
            Scenario {
                id: "s1".into(),
                description: "Office meeting planning between colleagues".into(),
                dialogues: vec![
                    DialogueTurn { speaker: "Alice".into(), content: "The Q3 review meeting is scheduled for next Friday at 2pm in Conference Room B.".into(), timestamp: Some("2025-06-10T09:00:00Z".into()) },
                    DialogueTurn { speaker: "Bob".into(), content: "Got it. I'll prepare the sales report. Should I also bring the customer feedback data?".into(), timestamp: Some("2025-06-10T09:05:00Z".into()) },
                    DialogueTurn { speaker: "Alice".into(), content: "Yes, please. Also invite Charlie from the Berlin office — he has insights on the European market.".into(), timestamp: Some("2025-06-10T09:10:00Z".into()) },
                    DialogueTurn { speaker: "Bob".into(), content: "Will do. I spoke with Charlie yesterday and he mentioned their new partnership with Siemens is going well.".into(), timestamp: Some("2025-06-10T09:15:00Z".into()) },
                    DialogueTurn { speaker: "Alice".into(), content: "Great news! Let's dedicate 10 minutes of the meeting to discuss the Siemens partnership.".into(), timestamp: Some("2025-06-10T09:20:00Z".into()) },
                ],
                questions: vec![
                    Question { id: "s1_q1".into(), question: "When is the Q3 review meeting?".into(), answer: "Next Friday at 2pm".into(), category: QuestionCategory::SingleHop, acceptable_answers: vec!["Friday at 2pm".into(), "next Friday at 2 PM".into()] },
                    Question { id: "s1_q2".into(), question: "Where is the meeting taking place?".into(), answer: "Conference Room B".into(), category: QuestionCategory::SingleHop, acceptable_answers: vec!["Room B".into(), "conference room B".into()] },
                    Question { id: "s1_q3".into(), question: "What will Bob prepare for the meeting?".into(), answer: "The sales report and customer feedback data".into(), category: QuestionCategory::MultiHop, acceptable_answers: vec!["sales report".into(), "sales report and customer feedback".into()] },
                    Question { id: "s1_q4".into(), question: "Which company did Charlie's office partner with?".into(), answer: "Siemens".into(), category: QuestionCategory::SingleHop, acceptable_answers: vec![] },
                    Question { id: "s1_q5".into(), question: "Where is Charlie based?".into(), answer: "Berlin".into(), category: QuestionCategory::SingleHop, acceptable_answers: vec!["the Berlin office".into()] },
                    Question { id: "s1_q6".into(), question: "Did Alice say the meeting would be on Monday?".into(), answer: "No, the meeting is on Friday".into(), category: QuestionCategory::Adversarial, acceptable_answers: vec!["No".into(), "No, it's on Friday".into()] },
                ],
            },
            Scenario {
                id: "s2".into(),
                description: "Multi-session travel planning".into(),
                dialogues: vec![
                    DialogueTurn { speaker: "Dana".into(), content: "I'm planning a trip to Tokyo in November. I want to visit Akihabara and try authentic ramen.".into(), timestamp: Some("2025-09-01T10:00:00Z".into()) },
                    DialogueTurn { speaker: "Eve".into(), content: "Nice! I went to Tokyo last year. Ichiran Ramen in Shibuya is amazing. Also, November is perfect for autumn leaves at Meiji Shrine.".into(), timestamp: Some("2025-09-01T10:05:00Z".into()) },
                    DialogueTurn { speaker: "Dana".into(), content: "Thanks for the tips! I'll book a hotel near Shibuya station then.".into(), timestamp: Some("2025-09-01T10:10:00Z".into()) },
                    DialogueTurn { speaker: "Dana".into(), content: "Update: I changed my plans. Going to Kyoto instead of Tokyo because my friend moved there.".into(), timestamp: Some("2025-10-15T14:00:00Z".into()) },
                    DialogueTurn { speaker: "Eve".into(), content: "Kyoto is beautiful in November too! Visit Fushimi Inari Shrine and the bamboo grove in Arashiyama.".into(), timestamp: Some("2025-10-15T14:05:00Z".into()) },
                ],
                questions: vec![
                    Question { id: "s2_q1".into(), question: "Where is Dana traveling in November?".into(), answer: "Kyoto".into(), category: QuestionCategory::Temporal, acceptable_answers: vec!["Kyoto, Japan".into()] },
                    Question { id: "s2_q2".into(), question: "Why did Dana change the travel destination?".into(), answer: "Because her friend moved to Kyoto".into(), category: QuestionCategory::SingleHop, acceptable_answers: vec!["friend moved to Kyoto".into(), "her friend moved there".into()] },
                    Question { id: "s2_q3".into(), question: "What ramen place did Eve recommend?".into(), answer: "Ichiran Ramen in Shibuya".into(), category: QuestionCategory::SingleHop, acceptable_answers: vec!["Ichiran Ramen".into(), "Ichiran".into()] },
                    Question { id: "s2_q4".into(), question: "Is Dana still planning to visit Akihabara?".into(), answer: "No, Dana changed plans to Kyoto instead of Tokyo".into(), category: QuestionCategory::MultiHop, acceptable_answers: vec!["No".into(), "No, she changed to Kyoto".into()] },
                    Question { id: "s2_q5".into(), question: "What shrines were recommended across both destinations?".into(), answer: "Meiji Shrine in Tokyo and Fushimi Inari Shrine in Kyoto".into(), category: QuestionCategory::MultiHop, acceptable_answers: vec!["Meiji Shrine and Fushimi Inari".into()] },
                ],
            },
        ],
    }
}
