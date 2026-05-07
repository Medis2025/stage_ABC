# Stage 4 Generated Output

This directory contains reproducible generated output from the Stage 4 pilot
export.

The current release-shaped output is:

```text
pilot_hpo_s3r_data/
```

Regenerate it with:

```bash
python Candidate/STAGE4/run_stage4_export.py export-all \
  --config Candidate/STAGE4/configs/pilot_export.json \
  --output Candidate/STAGE4/out/pilot_hpo_s3r_data

python Candidate/STAGE4/run_stage4_export.py release \
  --data-dir Candidate/STAGE4/out/pilot_hpo_s3r_data
```

The `RELEASED` marker means the generated pilot passed reference validation,
offset validation, quality checks, and manifest checksum checks.
