# Stage 4 Schemas

This directory contains JSON Schema files for the seven Stage ABC to S3R JSONL
tables:

- `nodes.schema.json`
- `evidence_chunks.schema.json`
- `span_supervision.schema.json`
- `retrieval_pairs.schema.json`
- `rerank_pairs.schema.json`
- `abstention_samples.schema.json`
- `graph_edges.schema.json`

The current schemas define required fields, core object shapes, and key enums.
They are intentionally permissive with `additionalProperties: true` so pilot
experiments can add metadata without breaking downstream validation.

Reference integrity and span offset checks are implemented in:

```text
Candidate/STAGE4/run_stage4_export.py
```
