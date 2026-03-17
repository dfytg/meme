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
