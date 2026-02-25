# Stage 3 – Phase A: Evidence Construction and Context Extraction

## Overview

**Stage 3 – Phase A** is designed to transform ontology-driven phenotype queries into a structured, reproducible evidence asset that can support downstream **Phase B corpus construction** and **model supervision** (e.g., S³R / HPO-MoRE training).

This stage performs **no model training**. Instead, it builds a multi-layered evidence representation by:

1. Generating PubMed search queries from HPO seeds.
2. Retrieving and caching PubMed abstracts (weak evidence).
3. Linking PubMed records to PMC full-text articles.
4. Parsing PMC XML into stable, addressable text units (chunks).
5. Extracting rule-based candidate spans (measurements, durations, physiological values).
6. Building HPO-level indexes and reusable evidence pools.

The outputs of Phase A are **resume-safe**, **cache-aware**, and **fully traceable**, forming a clean boundary between retrieval and generative supervision.

---

## Inputs

Phase A requires two JSONL inputs:

### 1. `queries_jsonl`

Each line represents an HPO seed definition and must include at least:

```json
{ "hpo_id": "HP:0001250", ... }
```

This file defines **which HPO concepts** will be processed.

---

### 2. `neighbors_jsonl`

Encodes HPO neighborhood or co-occurrence structure.

Used by `ClusterTermSampler` to generate multi-scale PubMed search queries (seed0 / seed1 / seed2 / negative variants).

---

## Output Directory Layout

Each execution creates a timestamped `run_dir` under the specified `--out_dir`.

Example:

```
out_phaseA/20260126_164618/
├── manifest.json
├── run_config.json
├── terms/
├── pubmed/
├── phaseA/
└── metrics/
```

---

## Core Outputs

### `manifest.json`

The canonical **entry point** for downstream stages. Contains paths to all critical assets and global statistics.

### `run_config.json`

Captures all runtime arguments and configuration for full reproducibility.

---

## Phase A Pipeline Description

### Part I — Term Sampling (Local, Offline)

**Outputs:**

* `terms/terms.jsonl`
* `terms/terms_selected.jsonl`

**Purpose:**
Generate a diverse set of PubMed search queries per HPO seed using ontology structure.

This step is deterministic, resume-safe, and does not perform network requests.

---

### Part II — PubMed Retrieval and Weak Evidence Construction

#### Step 2.1: PubMed ESearch (Query → PMIDs)

**Outputs:**

* `pubmed/query_to_pmids.jsonl`

Maps each generated query to a list of PubMed identifiers (PMIDs).

Optional LLM-based repair is applied **only** to true zero-hit queries.

---

#### Step 2.2: PubMed EFetch (PMID → Abstract)

**Outputs:**

* `pubmed/pmid_to_abstract.jsonl`

Each entry contains:

* PMID
* Title
* Abstract text
* Journal and publication year

Abstracts are normalized and cached as a **weak evidence layer**.

---

#### Step 2.3: Evidence Pool Construction

**Outputs:**

* `pubmed/evidence_pool.jsonl`
* `pubmed/by_hpo/<HPO>.jsonl`

Each record represents:

```json
{
  "hpo_id": "HP:...",
  "pmid": "...",
  "abstract": "...",
  "sha1_abstract_norm": "..."
}
```

This forms an HPO-conditioned abstract evidence pool used for prior modeling and conditional generation.

---

#### Step 2.4: PubMed → PMC Linking

**Outputs:**

* `pubmed/query_to_pmcids.jsonl`
* `metrics/pmcid_index.jsonl`
* `pubmed/hpo_to_pmids.jsonl`
* `pubmed/hpo_to_pmcids.jsonl`

Maps PubMed records to PMC full-text articles and builds **HPO-level indexes**.

---

### Part III — PMC Full-Text Parsing and Candidate Extraction

#### Step 3.1: PMC XML Retrieval

PMC full-text XML files are fetched and cached under:

```
<run_dir>/.cache/pmc_xml/
```

---

#### Step 3.2: Phase A Extraction (Per-PMCID)

For each PMCID:

**Outputs:**

```
phaseA/per_pmcid/<PMCID>/
├── case_chunks.jsonl
├── mentions_candidates.jsonl
└── DONE
```

* `case_chunks.jsonl`: Stable, addressable text units with `chunk_key`.
* `mentions_candidates.jsonl`: Rule-extracted candidate spans (values, units, context).
* `DONE`: Atomic completion marker enabling resume-safe execution.

---

#### Step 3.3: Global Merge

**Outputs:**

* `phaseA/merged/case_chunks.jsonl`
* `phaseA/merged/mentions_candidates.jsonl`

These files aggregate all per-PMCID outputs and are the **primary inputs for Phase B**.

---

## Metrics and Diagnostics

Located under `metrics/`:

* `failures.jsonl`: All recorded errors (network, parsing, extraction).
* `phaseA_status.jsonl`: Per-PMCID extraction statistics.
* `summary.json`: Aggregate counts and paths.

---

## Completion Criteria

A Phase A run is considered **complete** if:

* `manifest.json` reports `phaseA_fail_pmcids == 0`.
* Number of `DONE` markers equals `phaseA_ok_pmcids`.
* Merged outputs are non-empty.
* `evidence_pool.jsonl` is non-empty.

---

## Role in the Overall Pipeline

Phase A establishes a **clean separation** between retrieval and supervision:

* Phase A: *Evidence discovery, normalization, and structuring.*
* Phase B: *Conditional generation, span alignment, and corpus construction.*

The assets produced here are intentionally reusable across multiple modeling strategies and experimental designs.

---

## Recommended Downstream Usage (Phase B)

Phase B should consume only:

* `manifest.json`
* `pubmed/evidence_pool.jsonl`
* `pubmed/hpo_to_pmcids.jsonl`
* `phaseA/merged/case_chunks.jsonl`
* `phaseA/merged/mentions_candidates.jsonl`

No additional external retrieval is required.

---

*This document is upload-ready and intended to serve as a formal description of the Stage 3 Phase A evidence construction process.*
