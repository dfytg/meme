//! Integration test: verify that the real LOCOMO dataset can be loaded and parsed.

use std::collections::HashMap;
use std::path::Path;

#[test]
fn load_locomo10_dataset() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("3rdparty/locomo/data/locomo10.json");

    if !path.exists() {
        eprintln!("Skipping: locomo10.json not found at {}", path.display());
        return;
    }

    let dataset = meme_bench::dataset::load_locomo(&path).expect("failed to load locomo10.json");

    assert_eq!(dataset.name, "LOCOMO");
    assert_eq!(
        dataset.scenarios.len(),
        10,
        "LOCOMO should have 10 conversations"
    );

    let total_dialogues: usize = dataset.scenarios.iter().map(|s| s.dialogues.len()).sum();
    let total_questions: usize = dataset.scenarios.iter().map(|s| s.questions.len()).sum();

    assert!(
        total_dialogues > 100,
        "expected >100 dialogues, got {total_dialogues}"
    );
    assert!(
        total_questions > 50,
        "expected >50 questions, got {total_questions}"
    );

    println!("LOCOMO dataset loaded successfully:");
    println!("  Scenarios: {}", dataset.scenarios.len());
    println!("  Total dialogues: {total_dialogues}");
    println!("  Total questions: {total_questions}");

    for s in &dataset.scenarios {
        println!(
            "  [{}] {} — {} dialogues, {} questions",
            s.id,
            s.description,
            s.dialogues.len(),
            s.questions.len()
        );
        assert!(
            !s.dialogues.is_empty(),
            "scenario {} has no dialogues",
            s.id
        );
        assert!(
            !s.questions.is_empty(),
            "scenario {} has no questions",
            s.id
        );
    }

    let mut cat_counts: HashMap<String, usize> = HashMap::new();
    for s in &dataset.scenarios {
        for q in &s.questions {
            *cat_counts.entry(format!("{}", q.category)).or_default() += 1;
        }
    }
    println!("  Category distribution: {cat_counts:?}");
    assert!(
        cat_counts.len() >= 4,
        "expected at least 4 question categories"
    );
}
