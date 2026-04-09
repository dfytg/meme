//! Integration tests for `LanceDB`-backed `VectorStore`.

use arrow_array as _;
use arrow_schema as _;
use chrono as _;
#[cfg(feature = "onnx")]
use fastembed as _;
use futures as _;
use lancedb as _;
use regex as _;
use reqwest as _;
use rusqlite as _;
use serde as _;
use serde_json as _;
use thiserror as _;
use toml as _;
use tracing as _;
use tracing_subscriber as _;

#[cfg(test)]
mod tests {
    use meme::model::Memory;
    use meme::store::VectorStore;

    async fn temp_store(dim: usize) -> VectorStore {
        let dir = std::env::temp_dir().join(format!("meme_test_{}", uuid::Uuid::new_v4()));
        let path = dir.to_str().unwrap_or("meme_test_fallback");
        VectorStore::open(path, "test_memories", dim)
            .await
            .expect("failed to open test store")
    }

    fn dummy_entry(text: &str) -> Memory {
        let mut e = Memory::new(text);
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

        assert_eq!(store.count(None).await.unwrap(), 0);

        let entries = vec![dummy_entry("Alice met Bob at the park")];
        let vectors = vec![random_vec(8)];
        store.add_entries(&entries, &vectors).await.unwrap();

        assert_eq!(store.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn add_multiple_and_get_all() {
        let store = temp_store(8).await;

        let entries = vec![
            dummy_entry("First fact"),
            dummy_entry("Second fact"),
            dummy_entry("Third fact"),
        ];
        let vectors: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..8).map(|j| ((i * 8 + j) as f32 * 0.1).sin()).collect())
            .collect();

        store.add_entries(&entries, &vectors).await.unwrap();
        let all = store.get_all(None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn semantic_search_returns_results() {
        let store = temp_store(8).await;

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

        let results = store.semantic_search(&v1, 5, None).await.unwrap();
        assert!(!results.is_empty());
        let first = results.first().expect("non-empty results");
        assert_eq!(first.content, "The weather is sunny today");
    }

    #[tokio::test]
    async fn keyword_search_like_fallback() {
        let store = temp_store(8).await;

        let entries = vec![
            dummy_entry("Alice met Bob at Tokyo station"),
            dummy_entry("Charlie went to the gym"),
        ];
        let vectors: Vec<Vec<f32>> = vec![random_vec(8), random_vec(8)];
        store.add_entries(&entries, &vectors).await.unwrap();

        let results = store
            .keyword_search(&["Tokyo".into()], 5, None)
            .await
            .unwrap();
        assert!(
            results.iter().any(|e| e.content.contains("Tokyo")),
            "expected keyword match for Tokyo"
        );
    }

    #[tokio::test]
    async fn structured_search_by_persons() {
        let store = temp_store(8).await;

        let mut e1 = dummy_entry("Alice met Bob");
        e1.persons = vec!["Alice".into(), "Bob".into()];
        let mut e2 = dummy_entry("Charlie went home");
        e2.persons = vec!["Charlie".into()];

        store
            .add_entries(&[e1, e2], &[random_vec(8), random_vec(8)])
            .await
            .unwrap();

        let filter = meme::model::MetadataFilter {
            persons: Some(vec!["Bob".into()]),
            ..Default::default()
        };
        let results = store.structured_search(&filter, 5, None).await.unwrap();
        assert!(results.iter().any(|e| e.persons.contains(&"Bob".into())));
    }

    #[tokio::test]
    async fn delete_entries_by_id() {
        let store = temp_store(8).await;

        let entries = vec![dummy_entry("To be deleted"), dummy_entry("To be kept")];
        let id_to_delete = entries.first().expect("non-empty entries").id;
        let vectors = vec![random_vec(8), random_vec(8)];
        store.add_entries(&entries, &vectors).await.unwrap();
        assert_eq!(store.count(None).await.unwrap(), 2);

        store.delete_entries(&[id_to_delete]).await.unwrap();
        assert_eq!(store.count(None).await.unwrap(), 1);

        let remaining = store.get_all(None).await.unwrap();
        let first = remaining.first().expect("one remaining");
        assert_eq!(first.content, "To be kept");
    }

    #[tokio::test]
    async fn namespace_isolation() {
        let store = temp_store(8).await;

        let mut e1 = dummy_entry("User A data");
        e1.namespace = Some("user_a".into());
        let mut e2 = dummy_entry("User B data");
        e2.namespace = Some("user_b".into());

        store
            .add_entries(&[e1, e2], &[random_vec(8), random_vec(8)])
            .await
            .unwrap();

        assert_eq!(store.count(Some("user_a")).await.unwrap(), 1);
        assert_eq!(store.count(Some("user_b")).await.unwrap(), 1);
        assert_eq!(store.count(None).await.unwrap(), 2);

        let results_a = store.get_all(Some("user_a")).await.unwrap();
        let first = results_a.first().expect("user_a has one entry");
        assert_eq!(first.content, "User A data");
    }

    #[tokio::test]
    async fn namespace_clear() {
        let store = temp_store(8).await;

        let mut e1 = dummy_entry("User A data");
        e1.namespace = Some("user_a".into());
        let mut e2 = dummy_entry("User B data");
        e2.namespace = Some("user_b".into());

        store
            .add_entries(&[e1, e2], &[random_vec(8), random_vec(8)])
            .await
            .unwrap();

        store.clear(Some("user_a")).await.unwrap();

        assert_eq!(store.count(None).await.unwrap(), 1);
        let remaining = store.get_all(None).await.unwrap();
        let first = remaining.first().expect("one remaining");
        assert_eq!(first.content, "User B data");
    }

    #[tokio::test]
    async fn clear_all_removes_everything() {
        let store = temp_store(8).await;

        store
            .add_entries(&[dummy_entry("data")], &[random_vec(8)])
            .await
            .unwrap();
        assert_eq!(store.count(None).await.unwrap(), 1);

        store.clear_all().await.unwrap();
        assert_eq!(store.count(None).await.unwrap(), 0);
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
    async fn delete_nonexistent_uuid_succeeds() {
        let store = temp_store(8).await;
        let result = store.delete_entries(&[uuid::Uuid::new_v4()]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn get_all_with_vectors_roundtrip() {
        let store = temp_store(4).await;
        let v = vec![0.1, 0.2, 0.3, 0.4];
        store
            .add_entries(&[dummy_entry("roundtrip")], std::slice::from_ref(&v))
            .await
            .unwrap();

        let pairs = store.get_all_with_vectors(None).await.unwrap();
        assert_eq!(pairs.len(), 1);
        let first = pairs.first().expect("one pair");
        assert_eq!(first.0.content, "roundtrip");
        for (a, b) in first.1.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn update_entry_replaces_content() {
        let store = temp_store(4).await;
        let entry = dummy_entry("original text");
        let id = entry.id;
        let v = vec![0.1, 0.2, 0.3, 0.4];
        store
            .add_entries(&[entry], std::slice::from_ref(&v))
            .await
            .unwrap();

        let mut updated = store.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(updated.content, "original text");

        updated.content = "updated text".to_owned();
        let new_v = vec![0.5, 0.6, 0.7, 0.8];
        store.update_entry(&updated, &new_v).await.unwrap();

        let fetched = store.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(fetched.content, "updated text");
        assert_eq!(store.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn history_store_record_and_query() {
        let dir = std::env::temp_dir().join(format!("meme_hist_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("history.db");
        let store = meme::store::HistoryStore::open(&db_path).unwrap();

        let mem_id = uuid::Uuid::new_v4();
        store
            .record(mem_id, meme::EventType::Add, None, Some("hello"), None)
            .await
            .unwrap();
        store
            .record(
                mem_id,
                meme::EventType::Update,
                Some("hello"),
                Some("hello world"),
                None,
            )
            .await
            .unwrap();
        store
            .record(
                mem_id,
                meme::EventType::Delete,
                Some("hello world"),
                None,
                None,
            )
            .await
            .unwrap();

        let events = store.get_history(mem_id, None).await.unwrap();
        assert_eq!(events.len(), 3);
        let ev0 = events.first().expect("3 events");
        let ev1 = events.get(1).expect("3 events");
        let ev2 = events.get(2).expect("3 events");
        assert_eq!(ev0.event_type.as_str(), "add");
        assert_eq!(ev1.event_type.as_str(), "update");
        assert_eq!(ev2.event_type.as_str(), "delete");
        assert_eq!(ev0.new_content.as_deref(), Some("hello"));
        assert_eq!(ev1.old_content.as_deref(), Some("hello"));
        assert_eq!(ev1.new_content.as_deref(), Some("hello world"));

        let other_id = uuid::Uuid::new_v4();
        let empty = store.get_history(other_id, None).await.unwrap();
        assert!(empty.is_empty());
    }
}
