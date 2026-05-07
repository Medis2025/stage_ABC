# Stage ABC to S3R Training Data Contract v1.0

## Purpose

This document is the mandatory data contract for the Stage ABC export layer,
also called Stage 4. Stage ABC is an evidence-grounded data factory for S3R:

- S1 Router: span boundary and soft routing supervision.
- S2 Retriever: ontology-aware dense retrieval pairs.
- S3 Reranker: structured positive, negative, and abstention pairs.
- Graph-RAG: ontology, semantic, and evidence graph edges.

All Stage 4 outputs must be traceable, span-level, silver-label, and referentially
valid across files.

## Required Output Layout

```text
pilot_hpo_s3r_data/
├── raw_baseline_outputs/
├── processed/
│   ├── nodes.jsonl
│   ├── evidence_chunks.jsonl
│   ├── span_supervision.jsonl
│   ├── retrieval_pairs.jsonl
│   ├── rerank_pairs.jsonl
│   ├── abstention_samples.jsonl
│   └── graph_edges.jsonl
├── indexes/
│   ├── node_index.faiss
│   ├── evidence_index.faiss
│   └── index_manifest.json
├── manifests/
│   ├── baseline_manifest.json
│   ├── s3r_data_manifest.json
│   └── splits_manifest.json
└── validation/
    ├── referential_integrity_report.json
    └── data_quality_report.json
```

## Required JSONL Files

### `nodes.jsonl`

Ontology node table. Required fields:

- `node_id`
- `node_type`
- `name`
- `definition`
- `synonyms`
- `parents`
- `children`
- `embedding_text`
- `evidence_count`
- `tier`
- `uses_parent_evidence`
- `inherited_from`
- `ontology_subtree`
- `depth_in_ontology`
- `version_info`

Allowed `tier` values:

- `well_supported`
- `medium`
- `sparse`
- `parent_inherited`

Long-tail rule: nodes with `evidence_count=0` must set
`uses_parent_evidence=true` and point `inherited_from` to an existing node.

### `evidence_chunks.jsonl`

Evidence chunk table. Required fields:

- `evidence_id`
- `source_id`
- `pmcid`
- `section`
- `style`
- `style_source`
- `text`
- `char_length`
- `linked_hpo`
- `span_offsets`
- `evidence_level`
- `split`
- `split_strategy`
- `extraction_provenance`

Each `span_offsets` item must include:

- `start`
- `end`
- `text`
- `hpo_id`
- `extraction_method`
- `confidence_score`
- `confidence_tier`
- `tier_source`

### `span_supervision.jsonl`

S1 Router training data. Required fields:

- `sample_id`
- `source_evidence_id`
- `text`
- `char_length`
- `spans`
- `style`
- `split`
- `metadata`

Each span must include:

- `start`
- `end`
- `text`
- `label`
- `span_type`
- `confidence_tier`
- `expression_type`
- `source`

### `retrieval_pairs.jsonl`

S2 Retriever training data. Required fields:

- `sample_id`
- `source_evidence_id`
- `query_span`
- `span_start_in_context`
- `span_end_in_context`
- `query_context`
- `encoding_mode_hint`
- `positive_node_id`
- `positive_text`
- `positive_evidence_ids`
- `hard_negatives`
- `confidence_tier`
- `split`

Allowed hard negative types:

- `ontology_sibling`
- `ontology_parent`
- `ontology_child`
- `embedding_neighbor`
- `random`

### `rerank_pairs.jsonl`

S3 Reranker positive and negative pairs. Required fields:

- `sample_id`
- `source_evidence_id`
- `query`
- `span`
- `span_start`
- `span_end`
- `candidate_node_id`
- `candidate_name`
- `candidate_definition`
- `candidate_synonyms`
- `evidence_ids`
- `graph_features`
- `label`
- `negative_type`
- `reason`
- `confidence_tier`
- `split`

Allowed reranker negative types:

- `retriever_top_k_wrong`
- `ontology_sibling_wrong`
- `ontology_parent_child_wrong`
- `embedding_close_wrong`
- `random_wrong`

Positive rows must use `label=1` and `negative_type=null`.

### `abstention_samples.jsonl`

S3 abstention training data. Required fields:

- `sample_id`
- `source_evidence_id`
- `text`
- `span`
- `span_start`
- `span_end`
- `candidate_node_ids`
- `candidate_top_k_source`
- `expected_label`
- `abstention_reason`
- `construction_method`
- `confidence_tier`
- `split`

Allowed abstention reasons:

- `span_is_not_phenotype`
- `phenotype_but_all_candidates_wrong`
- `ambiguous_or_incomplete_span`
- `negated_phenotype`

### `graph_edges.jsonl`

Graph structure table. Required fields:

- `edge_id`
- `source_id`
- `target_id`
- `edge_layer`
- `edge_type`
- `score`
- `metadata`

Allowed edge layers:

- `ontology`
- `semantic`
- `evidence`

## Required Referential Integrity Checks

- Every HPO/node reference must exist in `nodes.jsonl`.
- Every evidence reference must exist in `evidence_chunks.jsonl`.
- `nodes.inherited_from`, when non-null, must exist in `nodes.jsonl`.
- `graph_edges.source_id` and `graph_edges.target_id` must exist in
  `nodes.jsonl`.
- Every span offset must exactly match the substring in its parent text.

Stage 4 export is not releasable unless reference validation and offset
validation both pass.

## Stage 4 CLI Contract

```bash
python Candidate/STAGE4/run_stage4_export.py validate-references --data-dir pilot_hpo_s3r_data/processed --report pilot_hpo_s3r_data/validation/referential_integrity_report.json
python Candidate/STAGE4/run_stage4_export.py validate-offsets --data-dir pilot_hpo_s3r_data/processed --report pilot_hpo_s3r_data/validation/offset_validation_report.json
python Candidate/STAGE4/run_stage4_export.py validate-all --data-dir pilot_hpo_s3r_data/processed --validation-dir pilot_hpo_s3r_data/validation
```

## One-Sentence Summary

Stage ABC must export seven mutually referential, versionable, traceable JSONL
files that serve S1 span routing, S2 retrieval, S3 reranking/abstention, and
Graph-RAG expansion.
