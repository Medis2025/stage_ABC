# Stage 4 Configs

This directory contains JSON configs consumed by `run_stage4_export.py`.

## `pilot_export.json`

`pilot_export.json` is a tiny deterministic pilot dataset. It contains the rows
that `export-all` writes into the seven Stage 4 JSONL files.

Run:

```bash
python Candidate/STAGE4/run_stage4_export.py export-all \
  --config Candidate/STAGE4/configs/pilot_export.json \
  --output Candidate/STAGE4/out/pilot_hpo_s3r_data
```

The config is intentionally small and synthetic. It is for validating the Stage
4 data shape, not for model training quality.
