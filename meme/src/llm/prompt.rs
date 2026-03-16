//! Prompt templates for LLM interactions across the pipeline.

use crate::model::MemoryEntry;

/// Collection of prompt templates used throughout the pipeline.
#[derive(Debug, Clone, Copy)]
pub struct Prompts;

impl Prompts {
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
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information

[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
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
```

Now process the above dialogues. Return ONLY the JSON array, no other explanations."#
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
            ctx.push_str(&format!("- {}\n", entry.restatement));
        }
        ctx
    }

    /// Build the query analysis prompt (extract structured info from a user query).
    #[must_use]
    pub fn query_analysis(query: &str) -> String {
        format!(
            r#"Analyze the following query and extract key information:

Query: {query}

Please extract:
1. keywords: List of keywords (names, places, topic words, etc.)
2. persons: Person names mentioned
3. time_expression: Time expression (if any)
4. location: Location (if any)
5. entities: Entities (companies, products, etc.)

Return in JSON format:
```json
{{
  "keywords": ["keyword1", "keyword2"],
  "persons": ["name1", "name2"],
  "time_expression": "time expression or null",
  "location": "location or null",
  "entities": ["entity1"]
}}
```

Return ONLY JSON, no other content."#
        )
    }

    /// Build the information requirements analysis prompt (Stage 3 planning).
    #[must_use]
    pub fn information_requirements(query: &str) -> String {
        format!(
            r#"Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {query}

Think step by step:
1. What type of question is this? (factual, temporal, relational, explanatory, etc.)
2. What key entities, events, or concepts need to be identified?
3. What relationships or connections need to be established?
4. What minimal set of information pieces would be sufficient to answer this question?

Return your analysis in JSON format:
```json
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2"],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high/medium/low"
    }}
  ],
  "relationships": ["relationship1", "relationship2"],
  "minimal_queries_needed": 2
}}
```

Focus on identifying the minimal essential information needed, not exhaustive details.
Return ONLY the JSON, no other text."#
        )
    }

    /// Build the targeted query generation prompt.
    #[must_use]
    pub fn targeted_queries(query: &str, plan_json: &str) -> String {
        format!(
            r#"Based on the information requirements analysis, generate the minimal set of targeted search queries needed to gather the required information.

Original Question: {query}

Information Requirements Analysis:
{plan_json}

Generate the minimal set of search queries that would efficiently gather all the required information.

Guidelines:
1. Always include the original query as one option
2. Generate only the minimal necessary queries (usually 1-3)
3. Each query should target a specific information requirement
4. Avoid redundant or overlapping queries
5. Focus on efficiency - fewer, more targeted queries are better

Return your response in JSON format:
```json
{{
  "reasoning": "Brief explanation of the query strategy",
  "queries": [
    "targeted query 1",
    "targeted query 2"
  ]
}}
```

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
    pub fn missing_info_queries(
        query: &str,
        context_str: &str,
        required_info_json: &str,
    ) -> String {
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

    /// Format memory entries as context string for answer generation.
    #[must_use]
    pub fn format_contexts(entries: &[MemoryEntry]) -> String {
        entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut parts = vec![format!("[Context {}]", i + 1)];
                parts.push(format!("Content: {}", e.restatement));
                if let Some(ts) = e.timestamp {
                    parts.push(format!("Time: {}", ts.format("%+")));
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
            })
            .collect::<Vec<_>>()
            .join("\n\n")
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
                    line.push_str(&format!(" | Time: {}", ts.format("%+")));
                }
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
