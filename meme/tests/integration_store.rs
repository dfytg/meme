//! Integration tests for LanceDB-backed VectorStore.

use meme::model::{MemoryEntry, MetadataFilter};
use meme::store::{Scope, VectorStore};

async fn temp_store(dim: usize) -> VectorStore {
    let dir = std::env::temp_dir().join(format!("meme_test_{}", uuid::Uuid::new_v4()));
    VectorStore::open(dir.to_str().unwrap(), "test_memories", dim)
        .await
        .expect("failed to open test store")
}

fn dummy_entry(text: &str) -> MemoryEntry {
    let mut e = MemoryEntry::new(text);
    e.keywords = vec!["test".into()];
    e.persons = vec!["Alice".into()];
    e
}

fn random_vec(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * 0.01).sin()).collect()
}

#[tokio::test]
async fn add_and_count() {
    let store = temp_store(8).await;
    let scope = Scope::default();

    assert_eq!(store.count(&scope).await.unwrap(), 0);

    let entries = vec![dummy_entry("Alice met Bob at the park")];
    let vectors = vec![random_vec(8)];
    store.add_entries(&entries, &vectors).await.unwrap();

    assert_eq!(store.count(&scope).await.unwrap(), 1);
}

#[tokio::test]
async fn add_multiple_and_get_all() {
    let store = temp_store(8).await;
    let scope = Scope::default();

    let entries = vec![
        dummy_entry("First fact"),
        dummy_entry("Second fact"),
        dummy_entry("Third fact"),
    ];
    let vectors: Vec<Vec<f32>> = (0..3)
        .map(|i| (0..8).map(|j| ((i * 8 + j) as f32 * 0.1).sin()).collect())
        .collect();

    store.add_entries(&entries, &vectors).await.unwrap();
    let all = store.get_all(&scope).await.unwrap();
    assert_eq!(all.len(), 3);
}

#[tokio::test]
async fn semantic_search_returns_results() {
    let store = temp_store(8).await;
    let scope = Scope::default();

    let entries = vec![
        dummy_entry("The weather is sunny today"),
        dummy_entry("Alice loves programming in Rust"),
    ];
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    store
        .add_entries(&entries, &[v1.clone(), v2])
        .await
        .unwrap();

    let results = store.semantic_search(&v1, 5, &scope).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].restatement, "The weather is sunny today");
}

#[tokio::test]
async fn keyword_search_like_fallback() {
    let store = temp_store(8).await;
    let scope = Scope::default();

    let entries = vec![
        dummy_entry("Alice met Bob at Tokyo station"),
        dummy_entry("Charlie went to the gym"),
    ];
    let vectors: Vec<Vec<f32>> = vec![random_vec(8), random_vec(8)];
    store.add_entries(&entries, &vectors).await.unwrap();

    let results = store
        .keyword_search(&["Tokyo".into()], 5, &scope)
        .await
        .unwrap();
    assert!(
        results.iter().any(|e| e.restatement.contains("Tokyo")),
        "expected keyword match for Tokyo"
    );
}

#[tokio::test]
async fn structured_search_by_persons() {
    let store = temp_store(8).await;
    let scope = Scope::default();

    let mut e1 = dummy_entry("Alice met Bob");
    e1.persons = vec!["Alice".into(), "Bob".into()];
    let mut e2 = dummy_entry("Charlie went home");
    e2.persons = vec!["Charlie".into()];

    store
        .add_entries(&[e1, e2], &[random_vec(8), random_vec(8)])
        .await
        .unwrap();

    let filter = MetadataFilter {
        persons: Some(vec!["Bob".into()]),
        ..Default::default()
    };
    let results = store.structured_search(&filter, 5, &scope).await.unwrap();
    assert!(results.iter().any(|e| e.persons.contains(&"Bob".into())));
}

#[tokio::test]
async fn delete_entries_by_id() {
    let store = temp_store(8).await;
    let scope = Scope::default();

    let entries = vec![dummy_entry("To be deleted"), dummy_entry("To be kept")];
    let vectors = vec![random_vec(8), random_vec(8)];
    store.add_entries(&entries, &vectors).await.unwrap();
    assert_eq!(store.count(&scope).await.unwrap(), 2);

    let id_to_delete = entries[0].id.to_string();
    store.delete_entries(&[id_to_delete]).await.unwrap();
    assert_eq!(store.count(&scope).await.unwrap(), 1);

    let remaining = store.get_all(&scope).await.unwrap();
    assert_eq!(remaining[0].restatement, "To be kept");
}

#[tokio::test]
async fn scoped_isolation() {
    let store = temp_store(8).await;

    let mut e1 = dummy_entry("User A data");
    e1.user_id = Some("user_a".into());
    let mut e2 = dummy_entry("User B data");
    e2.user_id = Some("user_b".into());

    store
        .add_entries(&[e1, e2], &[random_vec(8), random_vec(8)])
        .await
        .unwrap();

    let scope_a = Scope {
        user_id: Some("user_a".into()),
        session_id: None,
    };
    let scope_b = Scope {
        user_id: Some("user_b".into()),
        session_id: None,
    };

    assert_eq!(store.count(&scope_a).await.unwrap(), 1);
    assert_eq!(store.count(&scope_b).await.unwrap(), 1);
    assert_eq!(store.count(&Scope::default()).await.unwrap(), 2);

    let results_a = store.get_all(&scope_a).await.unwrap();
    assert_eq!(results_a[0].restatement, "User A data");
}

#[tokio::test]
async fn scoped_clear() {
    let store = temp_store(8).await;

    let mut e1 = dummy_entry("User A data");
    e1.user_id = Some("user_a".into());
    let mut e2 = dummy_entry("User B data");
    e2.user_id = Some("user_b".into());

    store
        .add_entries(&[e1, e2], &[random_vec(8), random_vec(8)])
        .await
        .unwrap();

    let scope_a = Scope {
        user_id: Some("user_a".into()),
        session_id: None,
    };
    store.clear(&scope_a).await.unwrap();

    assert_eq!(store.count(&Scope::default()).await.unwrap(), 1);
    let remaining = store.get_all(&Scope::default()).await.unwrap();
    assert_eq!(remaining[0].restatement, "User B data");
}

#[tokio::test]
async fn clear_all_removes_everything() {
    let store = temp_store(8).await;

    store
        .add_entries(&[dummy_entry("data")], &[random_vec(8)])
        .await
        .unwrap();
    assert_eq!(store.count(&Scope::default()).await.unwrap(), 1);

    store.clear_all().await.unwrap();
    assert_eq!(store.count(&Scope::default()).await.unwrap(), 0);
}

#[tokio::test]
async fn dimension_mismatch_rejected() {
    let store = temp_store(8).await;
    let entries = vec![dummy_entry("test")];
    let wrong_dim_vec = vec![vec![1.0, 2.0, 3.0]]; // dim=3, expected 8

    let result = store.add_entries(&entries, &wrong_dim_vec).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("dimension mismatch"), "got: {err}");
}

#[tokio::test]
async fn entries_vectors_length_mismatch_rejected() {
    let store = temp_store(8).await;
    let entries = vec![dummy_entry("a"), dummy_entry("b")];
    let vectors = vec![random_vec(8)]; // 1 vector for 2 entries

    let result = store.add_entries(&entries, &vectors).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("length mismatch"), "got: {err}");
}

#[tokio::test]
async fn delete_invalid_uuid_rejected() {
    let store = temp_store(8).await;
    let result = store.delete_entries(&["not-a-uuid".into()]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn get_all_with_vectors_roundtrip() {
    let store = temp_store(4).await;
    let v = vec![0.1, 0.2, 0.3, 0.4];
    store
        .add_entries(&[dummy_entry("roundtrip")], &[v.clone()])
        .await
        .unwrap();

    let pairs = store.get_all_with_vectors().await.unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0.restatement, "roundtrip");
    for (a, b) in pairs[0].1.iter().zip(v.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}
