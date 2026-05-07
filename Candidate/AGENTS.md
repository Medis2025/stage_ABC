# AGENTS.md - HPO-MoRE Candidate Pipeline

## Scope

These instructions apply to the `Candidate/` directory of the HPO-MoRE repository.

In this conversation, only files under:

`E:\HPO_MoRE\stage_ABC\Candidate`

may be modified.

Do not modify repository files outside `Candidate/` unless the user explicitly gives a new instruction.

## Pipeline Summary

The Candidate pipeline is a research-oriented, script-driven workflow for building HPO-centered literature evidence, semantic topology, and weakly supervised phenotype corpora.

The main flow is:

1. Stage 1 - Query construction
   - Reads enriched HPO records.
   - Uses LLM prompts to generate multi-scale query phrases.
   - Produces query JSONL files such as `queries.jsonl` and `queries_refilled.jsonl`.

2. Stage 2 - Embedding and topology freezing
   - Uses local Qwen3 embedding clients.
   - Builds phenotype embedding views from HPO names and generated query phrases.
   - Computes kNN neighbors and freezes topology outputs such as `neighbors.jsonl`, `neighbors_idx.npy`, and `neighbors_sim.npy`.

3. Stage 3 Phase A - Literature retrieval and evidence extraction
   - Samples PubMed queries from seed HPOs and topology neighbors.
   - Retrieves PubMed abstracts and links them to PMC full text.
   - Parses PMC XML into case chunks.
   - Extracts rule-based candidate mentions and measurements.
   - Writes run manifests, evidence pools, HPO indexes, merged case chunks, and merged candidate mentions.

4. Stage 3 Phase B / B1B2B3 - Supervision-oriented corpus construction
   - B1 uses constrained LLM extraction over PubMed/PMC evidence.
   - B2 builds embedding pools from definition, medium, and weak evidence layers.
   - B3 samples cluster-aware phenotype sets and generates gated clinical-style corpus items.

5. Stage 4 - S3R training data export
   - Converts Stage 1-3 artifacts into the mandatory S3R data contract.
   - Produces seven JSONL files plus indexes, manifests, validation reports, and release gates.
   - Treats Stage ABC as an evidence-grounded data factory for S1 Router, S2 Retriever, S3 Reranker, and Graph-RAG.

## Design Principles

- Preserve the staged architecture: query construction, embedding/topology, evidence retrieval, candidate extraction, evidence reformulation, corpus generation.
- Keep JSONL file contracts stable unless the user explicitly requests a schema change.
- Prefer deterministic post-processing, validation, filtering, and gating around any LLM output.
- Treat LLM calls as constrained helpers, not as hidden sources of state.
- Preserve resume-safety behavior, especially DONE markers, manifest files, run configs, caches, and append-only JSONL outputs.
- Avoid introducing agent-style orchestration or hidden side effects.

## Implementation Discipline

- Read the relevant script before editing it.
- Keep changes local to the stage being modified.
- Do not rewrite large scripts opportunistically.
- Do not remove historical outputs, caches, or generated artifacts unless explicitly requested.
- Do not hard-code new machine-specific absolute paths.
- Prefer CLI flags, config entries, or path arguments over embedded local paths.
- Use existing JSON/JSONL helper patterns when adding outputs.
- Preserve UTF-8 encoding for source files and data files.

## External Dependencies

The pipeline may depend on:

- DeepSeek-compatible chat API through `DEEPSEEK_API_KEY`.
- NCBI PubMed/PMC APIs, usually requiring an email and optional API key.
- Local Qwen3 embedding/reranker model directories.
- PyTorch, transformers, numpy, tqdm, requests.
- FAISS when available, with sklearn fallback in some scripts.

Do not assume these services or model paths are available locally. When adding runnable commands, document the required environment variables and path arguments.

## Safety Notes

- Network-facing scripts should keep polite NCBI behavior and existing rate limiting.
- Long-running generation scripts should preserve retry, resume, and failure logging behavior.
- Generated corpora should keep audit fields such as seed HPOs, sampled sets, coverage stats, anchors, attempts, and debug metadata.
- Existing committed outputs are useful forensic artifacts; treat them as data unless the user asks for cleanup.
- Stage 4 outputs must pass referential integrity and span offset validation before they are used for S3R training.

## Recommended Review Checklist

Before finishing a change under `Candidate/`, check:

- The edited stage still has a clear input and output contract.
- Existing CLI arguments still work.
- Resume behavior was not broken.
- JSONL output remains line-delimited valid JSON.
- Any new path is configurable.
- Any LLM-dependent behavior is logged and bounded.
- Any deterministic filter or sampler uses an explicit seed where appropriate.
