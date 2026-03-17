//! Evaluation metrics — token-level F1 score following LOCOMO methodology.

use std::collections::HashMap;

use crate::dataset::QuestionCategory;

/// Result of evaluating a single question.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QuestionResult {
    /// Question identifier.
    pub question_id: String,
    /// Question category.
    pub category: QuestionCategory,
    /// Question text.
    pub question: String,
    /// Expected (ground truth) answer.
    pub expected: String,
    /// Model-predicted answer.
    pub predicted: String,
    /// Token-level F1 score.
    pub f1: f64,
    /// Token-level precision.
    pub precision: f64,
    /// Token-level recall.
    pub recall: f64,
    /// Whether the prediction exactly matches the reference.
    pub exact_match: bool,
    /// LLM-as-Judge score (0.0 or 1.0). `None` if judge was not used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_judge_score: Option<f64>,
}

/// Aggregated results for a scenario or the entire benchmark.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AggregateMetrics {
    /// Total number of evaluated questions.
    pub total_questions: usize,
    /// Mean token-level F1 across all questions.
    pub mean_f1: f64,
    /// Mean token-level precision.
    pub mean_precision: f64,
    /// Mean token-level recall.
    pub mean_recall: f64,
    /// Fraction of exact-match predictions.
    pub exact_match_rate: f64,
    /// Mean LLM-as-Judge score (if available).
    pub mean_llm_judge: Option<f64>,
    /// Per-category breakdown.
    pub per_category: HashMap<QuestionCategory, CategoryMetrics>,
}

/// Metrics for a single question category.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CategoryMetrics {
    /// Number of questions in this category.
    pub count: usize,
    /// Mean token-level F1.
    pub mean_f1: f64,
    /// Mean token-level precision.
    pub mean_precision: f64,
    /// Mean token-level recall.
    pub mean_recall: f64,
    /// Mean LLM-as-Judge score (if available).
    pub mean_llm_judge: Option<f64>,
}

/// Compute token-level F1 score between predicted and reference answers.
///
/// Follows the LOCOMO/SQuAD methodology:
/// - Tokenize by whitespace + lowercase + strip punctuation
/// - Compute precision/recall over token overlap
#[must_use]
pub fn token_f1(predicted: &str, reference: &str) -> (f64, f64, f64) {
    let pred_tokens = normalize_tokens(predicted);
    let ref_tokens = normalize_tokens(reference);

    if pred_tokens.is_empty() && ref_tokens.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    if pred_tokens.is_empty() || ref_tokens.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let pred_set: std::collections::HashSet<&str> =
        pred_tokens.iter().map(String::as_str).collect();
    let ref_set: std::collections::HashSet<&str> = ref_tokens.iter().map(String::as_str).collect();
    let overlap = pred_set.intersection(&ref_set).count();

    let precision = overlap as f64 / pred_set.len() as f64;
    let recall = overlap as f64 / ref_set.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (f1, precision, recall)
}

/// Check if predicted answer matches reference (or any acceptable alternative).
#[must_use]
pub fn is_exact_match(predicted: &str, reference: &str, alternatives: &[String]) -> bool {
    let norm_pred = normalize_answer(predicted);
    norm_pred == normalize_answer(reference)
        || alternatives
            .iter()
            .any(|alt| norm_pred == normalize_answer(alt))
}

/// Compute the best F1 among the reference and all acceptable alternatives.
#[must_use]
pub fn best_f1(predicted: &str, reference: &str, alternatives: &[String]) -> (f64, f64, f64) {
    let mut best = token_f1(predicted, reference);
    for alt in alternatives {
        let scores = token_f1(predicted, alt);
        if scores.0 > best.0 {
            best = scores;
        }
    }
    best
}

/// Aggregate individual question results into summary metrics.
#[must_use]
pub fn aggregate(results: &[QuestionResult]) -> AggregateMetrics {
    let total = results.len();
    if total == 0 {
        return AggregateMetrics {
            total_questions: 0,
            mean_f1: 0.0,
            mean_precision: 0.0,
            mean_recall: 0.0,
            exact_match_rate: 0.0,
            mean_llm_judge: None,
            per_category: HashMap::new(),
        };
    }

    let mean_f1 = results.iter().map(|r| r.f1).sum::<f64>() / total as f64;
    let mean_precision = results.iter().map(|r| r.precision).sum::<f64>() / total as f64;
    let mean_recall = results.iter().map(|r| r.recall).sum::<f64>() / total as f64;
    let exact_match_rate = results.iter().filter(|r| r.exact_match).count() as f64 / total as f64;

    let mut by_cat: HashMap<QuestionCategory, Vec<&QuestionResult>> = HashMap::new();
    for r in results {
        by_cat.entry(r.category).or_default().push(r);
    }

    let per_category = by_cat
        .into_iter()
        .map(|(cat, items)| {
            let n = items.len();
            let judge_scores: Vec<f64> = items.iter().filter_map(|r| r.llm_judge_score).collect();
            let cat_metrics = CategoryMetrics {
                count: n,
                mean_f1: items.iter().map(|r| r.f1).sum::<f64>() / n as f64,
                mean_precision: items.iter().map(|r| r.precision).sum::<f64>() / n as f64,
                mean_recall: items.iter().map(|r| r.recall).sum::<f64>() / n as f64,
                mean_llm_judge: if judge_scores.is_empty() {
                    None
                } else {
                    Some(judge_scores.iter().sum::<f64>() / judge_scores.len() as f64)
                },
            };
            (cat, cat_metrics)
        })
        .collect();

    let all_judge: Vec<f64> = results.iter().filter_map(|r| r.llm_judge_score).collect();
    let mean_llm_judge = if all_judge.is_empty() {
        None
    } else {
        Some(all_judge.iter().sum::<f64>() / all_judge.len() as f64)
    };

    AggregateMetrics {
        total_questions: total,
        mean_f1,
        mean_precision,
        mean_recall,
        exact_match_rate,
        mean_llm_judge,
        per_category,
    }
}

fn normalize_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty() && !is_stopword(t))
        .collect()
}

fn normalize_token(token: &str) -> String {
    token
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn normalize_answer(text: &str) -> String {
    normalize_tokens(text).join(" ")
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "the"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
            | "will"
            | "would"
            | "could"
            | "should"
            | "may"
            | "might"
            | "shall"
            | "can"
            | "to"
            | "of"
            | "in"
            | "for"
            | "on"
            | "with"
            | "at"
            | "by"
            | "from"
            | "as"
            | "into"
            | "about"
            | "that"
            | "this"
            | "it"
            | "its"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f1_identical() {
        let (f1, p, r) = token_f1("Conference Room B", "Conference Room B");
        assert!((f1 - 1.0).abs() < 1e-9);
        assert!((p - 1.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn f1_partial_overlap() {
        let (f1, _, _) = token_f1("Room B on Friday", "Conference Room B");
        assert!(f1 > 0.0);
        assert!(f1 < 1.0);
    }

    #[test]
    fn f1_no_overlap() {
        let (f1, p, r) = token_f1("completely different answer", "Tokyo station");
        assert!((f1 - 0.0).abs() < 1e-9);
        assert!((p - 0.0).abs() < 1e-9);
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn f1_empty() {
        let (f1, _, _) = token_f1("", "");
        assert!((f1 - 1.0).abs() < 1e-9);

        let (f1, _, _) = token_f1("", "something");
        assert!((f1 - 0.0).abs() < 1e-9);

        let (f1, _, _) = token_f1("something", "");
        assert!((f1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn f1_case_insensitive() {
        let (f1, _, _) = token_f1("KYOTO japan", "Kyoto Japan");
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn f1_punctuation_stripped() {
        let (f1, _, _) = token_f1("Kyoto, Japan.", "Kyoto Japan");
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn exact_match_primary() {
        assert!(is_exact_match("Kyoto", "Kyoto", &[]));
        assert!(is_exact_match("kyoto", "Kyoto", &[]));
    }

    #[test]
    fn exact_match_alternative() {
        assert!(is_exact_match(
            "Kyoto Japan",
            "Tokyo",
            &["Kyoto Japan".into()]
        ));
    }

    #[test]
    fn exact_match_miss() {
        assert!(!is_exact_match("Paris", "Kyoto", &["Tokyo".into()]));
    }

    #[test]
    fn best_f1_uses_alternatives() {
        let (f1, _, _) = best_f1("Ichiran", "Ichiran Ramen in Shibuya", &["Ichiran".into()]);
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_empty() {
        let agg = aggregate(&[]);
        assert_eq!(agg.total_questions, 0);
    }

    #[test]
    fn stopwords_filtered() {
        let tokens = normalize_tokens("the cat is on the mat");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"on".to_string()));
        assert!(tokens.contains(&"cat".to_string()));
        assert!(tokens.contains(&"mat".to_string()));
    }
}
