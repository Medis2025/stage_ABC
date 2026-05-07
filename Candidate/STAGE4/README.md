# Stage 4 - S3R Training Data Export

Stage 4 converts Stage ABC artifacts into the S3R training data contract. It is
the release boundary between the HPO-MoRE candidate pipeline and downstream S3R
training.

This initial implementation is intentionally small:

- JSON Schema files for the seven required JSONL outputs.
- A config-driven pilot exporter in `run_stage4_export.py`.
- Working `export-all`, `validate-references`, `validate-offsets`,
  `validate-all`, `check-quality`, `write-manifest`, and `release` commands.
- JSONL pilot data under `examples/pilot_hpo_s3r_data/`.
- Reproducible release output under `out/pilot_hpo_s3r_data/`.

## Why Stage 4 Exists

Stage ABC is the offline evidence and candidate factory. S3R is the downstream
model framework:

- S1 decides where to look with span routing.
- S2 decides what to retrieve with dense ontology retrieval.
- S3 decides what is correct with structured reranking and abstention.

Stage 4 is the boundary between those systems. Its output must be stable enough
for training code to consume without guessing field meanings or silently
repairing broken references.

## Commands

Generate the pilot dataset from config:

```bash
python Candidate/STAGE4/run_stage4_export.py export-all \
  --config Candidate/STAGE4/configs/pilot_export.json \
  --output Candidate/STAGE4/out/pilot_hpo_s3r_data
```

Release the generated pilot dataset:

```bash
python Candidate/STAGE4/run_stage4_export.py release \
  --data-dir Candidate/STAGE4/out/pilot_hpo_s3r_data
```

Run all pilot validations:

```bash
python Candidate/STAGE4/run_stage4_export.py validate-all \
  --data-dir Candidate/STAGE4/examples/pilot_hpo_s3r_data/processed \
  --validation-dir Candidate/STAGE4/examples/pilot_hpo_s3r_data/validation
```

Validate references only:

```bash
python Candidate/STAGE4/run_stage4_export.py validate-references \
  --data-dir Candidate/STAGE4/examples/pilot_hpo_s3r_data/processed \
  --report Candidate/STAGE4/examples/pilot_hpo_s3r_data/validation/referential_integrity_report.json
```

Validate offsets only:

```bash
python Candidate/STAGE4/run_stage4_export.py validate-offsets \
  --data-dir Candidate/STAGE4/examples/pilot_hpo_s3r_data/processed \
  --report Candidate/STAGE4/examples/pilot_hpo_s3r_data/validation/offset_validation_report.json
```

Run quality gate only:

```bash
python Candidate/STAGE4/run_stage4_export.py check-quality \
  --data-dir Candidate/STAGE4/out/pilot_hpo_s3r_data
```

Refresh manifest and checksums:

```bash
python Candidate/STAGE4/run_stage4_export.py write-manifest \
  --data-dir Candidate/STAGE4/out/pilot_hpo_s3r_data \
  --config Candidate/STAGE4/configs/pilot_export.json
```

## Pilot Data

The pilot dataset uses a few HPO nodes and evidence chunks to exercise the
contract:

- `HP:0001250` Seizure
- `HP:0001252` Muscular hypotonia
- `HP:0004322` Short stature
- `HP:0002014` Diarrhea

It is not biologically complete. It exists to prove that the Stage 4 data shape,
reference checks, and offset checks work before real Stage 1-3 artifacts are
connected.

## Output Layout

`export-all` writes this release-shaped directory:

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
├── manifests/
│   ├── baseline_manifest.json
│   ├── pilot_export_config_snapshot.json
│   ├── s3r_data_manifest.json
│   └── splits_manifest.json
├── validation/
│   ├── referential_integrity_report.json
│   ├── offset_validation_report.json
│   ├── quality_gate_report.json
│   └── release_report.json
└── RELEASED
```

## Release Gates

`release` passes only when all checks are true:

- all seven JSONL files exist and are non-empty
- cross-file references are valid
- span offsets exactly match source text
- retrieval hard negative types cover all required categories
- abstention reasons cover all required categories
- manifest checksums match current files
- parent-inherited nodes, when present, point to valid source nodes

## Contract

The authoritative Stage ABC to S3R data contract is:

```text
Candidate/STAGE4_S3R_EXPORT_CONTRACT.md
```
