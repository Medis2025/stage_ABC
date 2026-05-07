# HPO-MoRE Candidate Pipeline

This directory contains the candidate-generation side of HPO-MoRE. It is a
script-driven research pipeline for building literature-grounded HPO evidence,
semantic topology, and S3R-ready training data.

The key idea is:

```text
HPO ontology
  -> query expansion
  -> PubMed / PMC evidence
  -> embedding topology
  -> candidate spans and synthetic evidence
  -> Stage 4 S3R export
```

## Directory Map

```text
Candidate/
├── STAGE1/                         # HPO query expansion
├── STAGE2/                         # embedding and topology freezing
├── STAGE3/                         # evidence retrieval and corpus generation
├── STAGE4/                         # S3R training data export layer
├── STAGE4_S3R_EXPORT_CONTRACT.md   # mandatory Stage ABC -> S3R data contract
├── qwen_clients.py                 # local Qwen3 embedding/reranker clients
├── embed_eval.py                   # embedding view evaluation
└── freeze_embedding_clusters.py    # frozen neighbor topology builder
```

## Stage Summary

### Stage 1 - Query Construction

Stage 1 reads enriched HPO records and generates multi-scale query phrases:

- exact / canonical phrases
- descriptive variants
- mechanism-oriented variants
- domain/literature phrases

These queries are used downstream for literature retrieval and candidate
evidence construction.

### Stage 2 - Embedding and Topology

Stage 2 uses Qwen3 embeddings to encode HPO terms and generated phrases. It
builds kNN neighborhoods and frozen semantic topology files such as
`neighbors.jsonl`, `neighbors_idx.npy`, and `neighbors_sim.npy`.

### Stage 3 - Evidence and Corpus Construction

Stage 3 retrieves PubMed/PMC evidence, extracts candidate spans and measurement
mentions, filters evidence, builds evidence pools, and generates cluster-aware
synthetic training corpora.

### Stage 4 - S3R Export

Stage 4 converts Stage ABC artifacts into the S3R training data contract:

- `nodes.jsonl`
- `evidence_chunks.jsonl`
- `span_supervision.jsonl`
- `retrieval_pairs.jsonl`
- `rerank_pairs.jsonl`
- `abstention_samples.jsonl`
- `graph_edges.jsonl`

It also writes manifests, checksums, validation reports, and a release marker.

## S3R Role

Stage ABC is the data factory for S3R:

- S1 Router learns span boundaries and soft token routing.
- S2 Retriever learns ontology-aware high-recall retrieval.
- S3 Reranker learns evidence-grounded discrimination and abstention.

The Stage 4 contract is the stable boundary between this pipeline and S3R
training code.

## Current Reproducible Smoke Test

Run the Stage 4 pilot export and release from this repository root:

```powershell
cd E:\HPO_MoRE\stage_ABC\Candidate

python .\STAGE4\run_stage4_export.py export-all `
  --config .\STAGE4\configs\pilot_export.json `
  --output .\STAGE4\out\pilot_hpo_s3r_data

python .\STAGE4\run_stage4_export.py release `
  --data-dir .\STAGE4\out\pilot_hpo_s3r_data
```

Expected result:

```json
{
  "passed": true,
  "validation": {
    "referential_integrity_passed": true,
    "offset_validation_passed": true,
    "data_quality_passed": true
  }
}
```

## Safety Notes

- Do not hard-code new local machine paths.
- Treat generated corpora as data artifacts; do not delete them unless the user
  explicitly asks for cleanup.
- LLM outputs must pass deterministic schema, reference, and offset validation
  before being used for S3R training.
- Stage 4 release is blocked unless validation and quality gates pass.
