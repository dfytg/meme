//! Prompt templates for LLM interactions across the pipeline.

use std::fmt::Write;

use crate::model::MemoryEntry;

/// Build the extraction prompt for Stage 1 (Semantic Structured Compression).
///
/// Converts a dialogue window into structured memory entries.
#[must_use]
pub fn extraction(dialogue_text: &str, context: &str) -> String {
    format!(
        r#"Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Temporal Anchoring**: Convert ALL relative time references to absolute dates using the dialogue timestamps as reference. E.g., if dialogue is dated "25 May 2023" and speaker says "last Sunday", compute the actual date. If dialogue says "next month", compute the actual month.
4. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
5. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information

[Output Format]
Return a JSON object containing an "entries" array:

```json
{{
  "entries": [
    {{
      "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
      "keywords": ["keyword1", "keyword2"],
      "timestamp": "YYYY-MM-DDTHH:MM:SSZ or null",
      "location": "location name or null",
      "persons": ["name1", "name2"],
      "entities": ["entity1", "entity2"],
      "topic": "topic phrase"
    }}
  ]
}}
```

Now process the above dialogues. Return ONLY the JSON object, no other explanations."#
    )
}

/// Build a prompt to re-extract structured fields from a single memory text.
///
/// Used by `update()` to keep metadata in sync after content changes.
#[must_use]
pub fn re_extract(text: &str) -> String {
    format!(
        r#"Extract structured metadata from the following memory text.

Text: {text}

Return a JSON object:
```json
{{
  "keywords": ["keyword1", "keyword2"],
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ or null",
  "location": "location name or null",
  "persons": ["name1", "name2"],
  "entities": ["entity1", "entity2"],
  "topic": "topic phrase or null"
}}
```

Return ONLY the JSON object."#
    )
}

/// Build the previous-window context string for extraction.
#[must_use]
pub fn extraction_context(previous_entries: &[MemoryEntry]) -> String {
    if previous_entries.is_empty() {
        return String::new();
    }
    let mut ctx =
        "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n".to_owned();
    for entry in previous_entries.iter().take(3) {
        let _ = writeln!(ctx, "- {}", entry.restatement);
    }
    ctx
}

/// Build the unified query plan prompt — combines query analysis and
/// information requirements into a single LLM call.
#[must_use]
pub fn query_plan(query: &str) -> String {
    format!(
        r#"Analyze the following question. Extract structured search metadata AND determine what information is needed to answer it.

Question: {query}

Return your analysis in JSON format:
```json
{{
  "keywords": ["keyword1", "keyword2"],
  "persons": ["name1", "name2"],
  "time_expression": "time expression or null",
  "location": "location or null",
  "entities": ["entity1"],
  "question_type": "factual/temporal/relational/explanatory",
  "required_info": ["what specific info is needed to answer"],
  "search_queries": ["targeted search query 1", "targeted search query 2"]
}}
```

Guidelines:
- keywords: Core topic words, names, places
- search_queries: 1-3 targeted queries to find the needed information (always include a variant of the original question)
- required_info: Minimal list of what facts are needed for a complete answer

Return ONLY the JSON, no other text."#
    )
}

/// Build the information completeness analysis prompt (reflection).
#[must_use]
pub fn completeness_check(query: &str, context_str: &str, required_info_json: &str) -> String {
    format!(
        r#"Analyze whether the provided information is sufficient to completely answer the original question, based on the identified information requirements.

Original Question: {query}

Required Information Types: {required_info_json}

Current Available Information:
{context_str}

Evaluate whether:
1. All required information types are addressed
2. The information is complete enough to provide a comprehensive answer
3. Any critical gaps remain that would prevent a satisfactory answer

Return your evaluation in JSON format:
```json
{{
  "assessment": "complete" or "incomplete",
  "reasoning": "Brief explanation of completeness assessment",
  "missing_info_types": ["list", "of", "missing", "types"],
  "coverage_percentage": 85
}}
```

Return ONLY the JSON, no other text."#
    )
}

/// Build the missing-info query generation prompt (reflection additional queries).
#[must_use]
pub fn missing_info_queries(query: &str, context_str: &str, required_info_json: &str) -> String {
    format!(
        r#"Based on the original question, required information types, and currently available information, generate targeted search queries to find the missing information.

Original Question: {query}

Required Information Types: {required_info_json}

Currently Available Information:
{context_str}

Generate 1-3 specific search queries that would help find the missing information.

Return your response in JSON format:
```json
{{
  "missing_analysis": "Brief analysis of what specific information is missing",
  "targeted_queries": [
    "specific query 1 for missing info",
    "specific query 2 for missing info"
  ]
}}
```

Return ONLY the JSON, no other text."#
    )
}

/// Build the answer generation prompt.
#[must_use]
pub fn answer(query: &str, context_str: &str) -> String {
    format!(
        r#"Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Now answer the question. Return ONLY the JSON, no other text."#
    )
}

/// Build the memory reconciliation prompt.
///
/// Given new extracted facts and existing similar memories, the LLM determines
/// the correct lifecycle action (ADD / UPDATE / DELETE / NOOP) for each new fact.
#[must_use]
pub fn reconcile(new_facts: &[&str], existing_memories: &[(usize, &str)]) -> String {
    let mut new_block = String::new();
    for (i, fact) in new_facts.iter().enumerate() {
        let _ = writeln!(new_block, "[New {i}] {fact}");
    }
    let mut existing_block = String::new();
    for (idx, text) in existing_memories {
        let _ = writeln!(existing_block, "[Existing {idx}] {text}");
    }
    format!(
        r#"You are a smart memory manager. Compare newly extracted facts with existing memories and decide the correct action for each new fact.

[New Facts]
{new_block}
[Existing Memories]
{existing_block}
For each new fact, choose ONE action:
- "add": New information not present in existing memories. Keep it.
- "update": Supersedes or corrects an existing memory. Specify which existing_index to replace.
- "delete": Contradicts an existing memory, making it obsolete. Specify which existing_index to remove.
- "noop": Already present in existing memories (duplicate). Skip it.

Guidelines:
1. If new fact updates an existing memory with more recent/accurate info, use "update".
2. If new fact directly contradicts an existing memory, use "delete" on the old one AND "add" the new.
3. If new fact conveys the same meaning as an existing one, use "noop".
4. If new fact is genuinely new information, use "add".

Return a JSON object:
```json
{{
  "actions": [
    {{"new_index": 0, "action": "add|update|delete|noop", "existing_index": null, "reason": "brief explanation"}}
  ]
}}
```

Return ONLY the JSON object."#
    )
}

/// Format memory entries as context string for answer generation.
#[must_use]
pub fn format_contexts(entries: &[&MemoryEntry]) -> String {
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| format_single_context(i, e))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_single_context(i: usize, e: &MemoryEntry) -> String {
    let mut parts = vec![format!("[Context {}]", i + 1)];
    parts.push(format!("Content: {}", e.restatement));
    if let Some(ts) = e.timestamp {
        parts.push(format!("Time: {}", ts.format("%d %B %Y %H:%M")));
    }
    if let Some(loc) = &e.location {
        parts.push(format!("Location: {loc}"));
    }
    if !e.persons.is_empty() {
        parts.push(format!("Persons: {}", e.persons.join(", ")));
    }
    if !e.entities.is_empty() {
        parts.push(format!("Related Entities: {}", e.entities.join(", ")));
    }
    if let Some(topic) = &e.topic {
        parts.push(format!("Topic: {topic}"));
    }
    parts.join("\n")
}

/// Format entries compactly for reflection/completeness checks.
#[must_use]
pub fn format_contexts_compact(entries: &[MemoryEntry]) -> String {
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut line = format!("[Info {}] {}", i + 1, e.restatement);
            if let Some(ts) = e.timestamp {
                let _ = write!(line, " | Time: {}", ts.format("%+"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}
