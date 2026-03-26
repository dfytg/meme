//! Benchmark runner — feeds dialogues into meme and evaluates QA performance.

use std::time::Instant;

use crate::dataset::Scenario;
use crate::metrics::{self, AggregateMetrics, QuestionResult};

/// Run command arguments.
#[derive(Debug, clap::Args)]
pub struct RunCmd {
    /// Path to the LOCOMO-format benchmark JSON file.
    #[arg(short, long)]
    pub dataset: String,

    /// LLM model to use.
    #[arg(short, long, default_value = "gpt-4.1-mini")]
    pub model: String,

    /// Output results to a JSON file.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Only run scenarios matching this ID prefix.
    #[arg(long)]
    pub filter: Option<String>,

    /// Maximum number of scenarios to run.
    #[arg(short, long)]
    pub limit: Option<usize>,

    /// Maximum number of questions per scenario.
    #[arg(long)]
    pub questions_limit: Option<usize>,
}

/// Full benchmark report.
#[derive(Debug, serde::Serialize)]
struct BenchmarkReport {
    dataset_name: String,
    model: String,
    total_scenarios: usize,
    total_questions: usize,
    duration_secs: f64,
    aggregate: AggregateMetrics,
    scenario_results: Vec<ScenarioReport>,
}

/// Per-scenario report.
#[derive(Debug, serde::Serialize)]
struct ScenarioReport {
    scenario_id: String,
    description: String,
    num_dialogues: usize,
    num_questions: usize,
    aggregate: AggregateMetrics,
    questions: Vec<QuestionResult>,
}

impl RunCmd {
    /// Execute the benchmark.
    ///
    /// # Errors
    ///
    /// Returns an error if dataset loading, meme initialization, or evaluation fails.
    pub async fn run(&self) -> anyhow::Result<()> {
        let path = std::path::Path::new(&self.dataset);
        let dataset = if let Ok(ds) = crate::dataset::load_locomo(path) {
            println!("Loaded LOCOMO-format dataset");
            ds
        } else {
            let content = std::fs::read_to_string(path)?;
            serde_json::from_str(&content)?
        };

        println!("=== meme LOCOMO Benchmark ===");
        println!(
            "Dataset: {} ({} scenarios)",
            dataset.name,
            dataset.scenarios.len()
        );
        println!("Model: {}", self.model);
        println!();

        let mut scenarios: Vec<&Scenario> = if let Some(filter) = &self.filter {
            dataset
                .scenarios
                .iter()
                .filter(|s| s.id.starts_with(filter))
                .collect()
        } else {
            dataset.scenarios.iter().collect()
        };

        if let Some(limit) = self.limit {
            scenarios.truncate(limit);
        }

        let t0 = Instant::now();
        let mut all_results: Vec<QuestionResult> = Vec::new();
        let mut scenario_reports = Vec::new();

        for (i, scenario) in scenarios.iter().enumerate() {
            println!(
                "[{}/{}] Scenario: {} — {}",
                i + 1,
                scenarios.len(),
                scenario.id,
                scenario.description
            );
            println!(
                "  {} dialogues, {} questions",
                scenario.dialogues.len(),
                scenario.questions.len()
            );

            match self.run_scenario(scenario).await {
                Ok((results, scenario_agg)) => {
                    println!(
                        "  F1: {:.1}%  EM: {:.1}%",
                        scenario_agg.mean_f1 * 100.0,
                        scenario_agg.exact_match_rate * 100.0
                    );
                    for (cat, cm) in &scenario_agg.per_category {
                        println!("    {cat}: F1={:.1}% (n={})", cm.mean_f1 * 100.0, cm.count);
                    }
                    scenario_reports.push(ScenarioReport {
                        scenario_id: scenario.id.clone(),
                        description: scenario.description.clone(),
                        num_dialogues: scenario.dialogues.len(),
                        num_questions: scenario.questions.len(),
                        aggregate: scenario_agg,
                        questions: results.clone(),
                    });
                    all_results.extend(results);
                }
                Err(e) => {
                    eprintln!("  ERROR: {e}");
                }
            }
            println!();
        }

        let duration = t0.elapsed().as_secs_f64();
        let aggregate = metrics::aggregate(&all_results);

        println!("=== Overall Results ===");
        println!("Questions: {}", aggregate.total_questions);
        println!("Mean F1: {:.1}%", aggregate.mean_f1 * 100.0);
        println!("Mean Precision: {:.1}%", aggregate.mean_precision * 100.0);
        println!("Mean Recall: {:.1}%", aggregate.mean_recall * 100.0);
        println!("Exact Match: {:.1}%", aggregate.exact_match_rate * 100.0);
        println!();
        println!("Per-category:");
        let mut cats: Vec<_> = aggregate.per_category.iter().collect();
        cats.sort_by_key(|(c, _)| format!("{c}"));
        for (cat, cm) in &cats {
            println!(
                "  {cat}: F1={:.1}%  P={:.1}%  R={:.1}%  (n={})",
                cm.mean_f1 * 100.0,
                cm.mean_precision * 100.0,
                cm.mean_recall * 100.0,
                cm.count
            );
        }
        println!();
        println!("Duration: {duration:.1}s");

        if let Some(output) = &self.output {
            let report = BenchmarkReport {
                dataset_name: dataset.name.clone(),
                model: self.model.clone(),
                total_scenarios: scenarios.len(),
                total_questions: all_results.len(),
                duration_secs: duration,
                aggregate,
                scenario_results: scenario_reports,
            };
            let json = serde_json::to_string_pretty(&report)?;
            std::fs::write(output, json)?;
            println!("Report written to {output}");
        }

        Ok(())
    }

    async fn run_scenario(
        &self,
        scenario: &Scenario,
    ) -> anyhow::Result<(Vec<QuestionResult>, AggregateMetrics)> {
        let meme = meme::MemeBuilder::new()
            .model(&self.model)
            .clear_db(true)
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let dialogues: Vec<meme::model::Dialogue> = scenario
            .dialogues
            .iter()
            .map(|d| {
                let mut dial = meme::model::Dialogue::new(&d.speaker, &d.content);
                if let Some(ts) = &d.timestamp
                    && let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts)
                {
                    dial = dial.with_timestamp(dt.with_timezone(&chrono::Utc));
                }
                dial
            })
            .collect();

        meme.add_dialogues(dialogues)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        meme.flush().await.map_err(|e| anyhow::anyhow!("{e}"))?;

        let stored = meme.count().await.map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("  Stored {stored} memory entries");

        let mut results = Vec::new();
        let questions: Vec<_> = self.questions_limit.map_or_else(
            || scenario.questions.iter().collect(),
            |ql| scenario.questions.iter().take(ql).collect(),
        );
        for q in questions {
            let predicted = meme
                .ask(&q.question)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let (f1, precision, recall) =
                metrics::best_f1(&predicted, &q.answer, &q.acceptable_answers);
            let exact_match = metrics::is_exact_match(&predicted, &q.answer, &q.acceptable_answers);

            results.push(QuestionResult {
                question_id: q.id.clone(),
                category: q.category,
                question: q.question.clone(),
                expected: q.answer.clone(),
                predicted: predicted.clone(),
                f1,
                precision,
                recall,
                exact_match,
                llm_judge_score: None,
            });

            let status = if f1 >= 0.8 {
                "✓"
            } else if f1 >= 0.3 {
                "~"
            } else {
                "✗"
            };
            println!(
                "  {status} [{:?}] Q: {} → F1={:.0}%",
                q.category,
                q.question,
                f1 * 100.0
            );
        }

        let agg = metrics::aggregate(&results);
        Ok((results, agg))
    }
}
