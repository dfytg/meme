//! Memory reconciliation — deduplicates and resolves conflicts between new and
//! existing memories via a single LLM call.

use std::collections::{HashMap, HashSet};

use futures::future;
use uuid::Uuid;

use crate::error::Result;
use crate::llm::{self, ChatOptions, LlmClient, Message, ReconcileResponse};
use crate::model::{Memory, Scope};
use crate::store::VectorStore;

/// Reconcile new entries against existing memories using the LLM.
///
/// For each new entry, finds the closest existing memories via ANN search,
/// then asks the LLM to decide per-entry: **add**, **update**, **delete**, or **noop**.
///
/// Returns `(entries_to_add, their_vectors, entries_to_delete_with_old_content)`.
///
/// # Errors
///
/// Returns an error if ANN search or LLM reconciliation fails.
pub async fn reconcile(
    llm: &LlmClient,
    store: &VectorStore,
    scope: &Scope,
    entries: &[Memory],
    vectors: &[Vec<f32>],
) -> Result<(Vec<Memory>, Vec<Vec<f32>>, Vec<(Uuid, String)>)> {
    let similarity_top_k = 5;

    let new_facts: Vec<&str> = entries.iter().map(|e| e.content.as_str()).collect();

    let ann_futures: Vec<_> = vectors
        .iter()
        .map(|vec_i| store.semantic_search(vec_i, similarity_top_k, scope))
        .collect();
    let all_existing: Vec<Vec<Memory>> = future::try_join_all(ann_futures).await?;

    let mut existing_map: HashMap<Uuid, (usize, String)> = HashMap::new();
    for group in &all_existing {
        for entry in group {
            let next_idx = existing_map.len();
            existing_map
                .entry(entry.id)
                .or_insert_with(|| (next_idx, entry.content.clone()));
        }
    }

    if existing_map.is_empty() {
        return Ok((entries.to_vec(), vectors.to_vec(), Vec::new()));
    }

    let existing_indexed: Vec<(usize, &str)> = {
        let mut v: Vec<(usize, &str)> = existing_map
            .values()
            .map(|(idx, text)| (*idx, text.as_str()))
            .collect();
        v.sort_by_key(|(idx, _)| *idx);
        v
    };

    let idx_to_uuid: HashMap<usize, Uuid> = existing_map
        .iter()
        .map(|(uid, (idx, _))| (*idx, *uid))
        .collect();

    let prompt_text = llm::prompt::reconcile(&new_facts, &existing_indexed);
    let messages = vec![
        Message::system("You are a smart memory manager. You must output valid JSON format."),
        Message::user(prompt_text),
    ];
    let opts = ChatOptions {
        temperature: 0.1,
        json_mode: true,
    };

    let resp: ReconcileResponse = llm.chat_structured(&messages, &opts).await?;

    let mut accepted = Vec::new();
    let mut accepted_vecs = Vec::new();
    let mut deletes: Vec<(Uuid, String)> = Vec::new();

    for action in &resp.actions {
        let Some(new_idx) = action.new_index else {
            continue;
        };
        if new_idx >= entries.len() {
            continue;
        }

        let target = action.existing_index.and_then(|eidx| {
            idx_to_uuid
                .get(&eidx)
                .and_then(|uid| existing_map.get(uid).map(|(_, text)| (*uid, text.clone())))
        });

        match action.action.as_str() {
            "update" => {
                if let Some(pair) = target {
                    deletes.push(pair);
                }
                accepted.push(entries[new_idx].clone());
                accepted_vecs.push(vectors[new_idx].clone());
            }
            "delete" => {
                if let Some(pair) = target {
                    deletes.push(pair);
                }
            }
            "noop" | "duplicate" => {}
            _ => {
                accepted.push(entries[new_idx].clone());
                accepted_vecs.push(vectors[new_idx].clone());
            }
        }
    }

    let handled: HashSet<usize> = resp.actions.iter().filter_map(|a| a.new_index).collect();
    for (i, entry) in entries.iter().enumerate() {
        if !handled.contains(&i) {
            accepted.push(entry.clone());
            accepted_vecs.push(vectors[i].clone());
        }
    }

    deletes.sort_by_key(|(uid, _)| *uid);
    deletes.dedup_by_key(|(uid, _)| *uid);

    Ok((accepted, accepted_vecs, deletes))
}
