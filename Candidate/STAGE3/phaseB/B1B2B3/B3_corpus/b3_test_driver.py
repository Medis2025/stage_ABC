#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b3_test_driver.py  (PLAN A+++: set-level corpus item = one "patient phenotype set")

REVISION (2026-02-04)
====================
This revision keeps your overall structure but fixes the *real* failure modes you observed:

A) Coverage accounting bug fix
   - Your debug["covered_hpos"] was truncated to 50 items for preview, then reused for coverage_ok.
   - Now we store BOTH:
       * covered_hpos_all (full)
       * covered_hpos_preview (first 50)
   - coverage_ratio_valid is computed from covered_hpos_all.

B) Targeted repair on missing phenotypes (attempt 3)
   - Instead of a generic "FINAL ATTEMPT", attempt 3 becomes a *missing-HPO repair*:
     It forces the model to mention missing phenotypes at least once while still satisfying per-line constraints.

C) Anchor quality improvements (reduce false hits / “generic token” inflation)
   - Do NOT use ambiguous anchors like "urate"/"uric acid" alone (caused hypo/hyper confusion).
   - Filter overly-generic one-token anchors (e.g., "total", "level", "bone", "epiphysis", "igg").
   - Keep polarity tokens (hyper/hypo/increased/decreased/chronic/transient) instead of dropping them.

D) Diversity check (lightweight)
   - Detect low diversity (duplicate 3-word starts / near-duplicate sentences).
   - If diversity is bad and we still have retries left, we nudge via prompt (attempt 2+).

Outputs remain the same:
  - B3_run_config.json
  - sets_sampled.jsonl
  - corpus_generated.jsonl
  - summary.json
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

from tqdm import tqdm
import numpy as np

from Clients.llm_client import LLMClient


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def read_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(p: str) -> Iterable[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_json(p: str, obj: Any) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(p: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------
# Domain data loaders
# ---------------------------------------------------------------------

def load_hpo_names(def_source_jsonl: str) -> Dict[str, str]:
    h2n: Dict[str, str] = {}
    for obj in iter_jsonl(def_source_jsonl):
        hid = obj.get("hpo_id")
        name = obj.get("name", "")
        if isinstance(hid, str) and hid.startswith("HP:"):
            h2n[hid] = str(name)
    return h2n

def load_master_ids(recluster_dir: str) -> List[str]:
    p = os.path.join(recluster_dir, "master_hpo_ids.json")
    obj = read_json(p)
    if isinstance(obj, list):
        ids = [x for x in obj if isinstance(x, str) and x.startswith("HP:")]
        if not ids:
            raise ValueError(f"master_hpo_ids.json list contains no HP: ids: {p}")
        return ids
    if isinstance(obj, dict) and "ids" in obj and isinstance(obj["ids"], list):
        ids = [x for x in obj["ids"] if isinstance(x, str) and x.startswith("HP:")]
        if ids:
            return ids
    raise ValueError(f"Unrecognized master_hpo_ids.json format: {p}")

def load_knn_round2(recluster_dir: str) -> np.ndarray:
    p = os.path.join(recluster_dir, "knn_round2_idx.npy")
    idx = np.load(p)
    if idx.ndim != 2:
        raise ValueError(f"knn_round2_idx.npy must be 2D, got {idx.shape}")
    return idx.astype(np.int32, copy=False)

def load_labels_round2(recluster_dir: str) -> np.ndarray:
    p = os.path.join(recluster_dir, "labels_round2.npy")
    lab = np.load(p)
    if lab.ndim != 1:
        raise ValueError(f"labels_round2.npy must be 1D, got {lab.shape}")
    return lab.astype(np.int32, copy=False)

def load_v1_smooth(recluster_dir: str) -> Optional[np.ndarray]:
    p = os.path.join(recluster_dir, "V1_smooth.npy")
    if not os.path.isfile(p):
        return None
    V = np.load(p).astype(np.float32, copy=False)
    return V


# ---------------------------------------------------------------------
# Evidence (lightweight test-mode)
# ---------------------------------------------------------------------

def build_pubmed_pool_index_by_hpo(
    evidence_pool_jsonl: str,
    max_items_per_hpo: int = 1
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for obj in iter_jsonl(evidence_pool_jsonl):
        hid = obj.get("hpo_id")
        if not (isinstance(hid, str) and hid.startswith("HP:")):
            continue
        arr = out.setdefault(hid, [])
        if len(arr) >= max_items_per_hpo:
            continue
        arr.append({
            "pmid": obj.get("pmid"),
            "title": obj.get("title", ""),
            "abstract": obj.get("abstract", ""),
            "year": obj.get("year_int", obj.get("year", "")),
        })
    return out

def build_case_chunks_cache(case_chunks_jsonl: str, max_keep: int = 200000) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    n = 0
    for obj in iter_jsonl(case_chunks_jsonl):
        chunks.append(obj)
        n += 1
        if max_keep and n >= max_keep:
            break
    return chunks

def collect_evidence_for_set(
    hpos: List[str],
    *,
    h2n: Dict[str, str],
    pubmed_by_hpo: Dict[str, List[Dict[str, Any]]],
    case_chunks: Optional[List[Dict[str, Any]]] = None,
    max_items_per_hpo: int = 1,
    max_chars: int = 1400,
) -> str:
    parts: List[str] = []
    parts.append("PHENOTYPE_SET:")
    for hid in hpos:
        parts.append(f"- {hid} | {h2n.get(hid,'')}".strip())

    def _fmt_pub(ev: Dict[str, Any]) -> str:
        title = str(ev.get("title", "")).strip()
        abstract = str(ev.get("abstract", "")).strip()
        pmid = str(ev.get("pmid", "")).strip()
        year = str(ev.get("year", "")).strip()
        return f"[PMID {pmid} | {year}] {title}\n{abstract}".strip()

    any_pub = False
    if max_items_per_hpo > 0:
        for hid in hpos:
            evs = pubmed_by_hpo.get(hid, [])[:max_items_per_hpo]
            if not evs:
                continue
            if not any_pub:
                parts.append("\nPUBMED_EVIDENCE:")
                any_pub = True
            parts.append(f"\n# Evidence for {hid} | {h2n.get(hid,'')}".strip())
            for ev in evs:
                parts.append(_fmt_pub(ev))

    if case_chunks:
        key = "|".join(hpos)
        idx = (abs(hash(key)) % len(case_chunks))
        ck = case_chunks[idx]
        txt = str(ck.get("text", "")).strip()
        if txt:
            parts.append("\nPMC_CASE_CHUNK_EXAMPLE:")
            parts.append(f"[{ck.get('chunk_key','')}] {txt}")

    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n\n[TRUNCATED]"
    return s


# ---------------------------------------------------------------------
# PLAN A: Set sampling (seed + K-1 neighbors)
# ---------------------------------------------------------------------

def sample_sets_from_knn(
    master_ids: List[str],
    knn_idx: np.ndarray,
    labels: np.ndarray,
    *,
    n_sets: int,
    k_min: int,
    k_max: int,
    per_seed_candidates: int,
    same_cluster_ratio: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    N = len(master_ids)
    if knn_idx.shape[0] != N:
        raise ValueError(f"knn_idx N mismatch: {knn_idx.shape[0]} vs {N}")
    if labels.shape[0] != N:
        raise ValueError(f"labels N mismatch: {labels.shape[0]} vs {N}")

    k_min = max(2, int(k_min))
    k_max = max(k_min, int(k_max))
    per_seed_candidates = max(5, int(per_seed_candidates))

    sets: List[Dict[str, Any]] = []
    used_signatures = set()

    seed_pool = rng.permutation(np.arange(N)).tolist()

    for si in seed_pool:
        if len(sets) >= n_sets:
            break

        K = int(rng.randint(k_min, k_max + 1))
        h_seed = master_ids[int(si)]
        c_seed = int(labels[int(si)])

        neigh = knn_idx[int(si)].tolist()
        neigh = neigh[:min(len(neigh), per_seed_candidates)]
        rng.shuffle(neigh)

        chosen_idx = [int(si)]
        chosen_set = {int(si)}

        same_list = [j for j in neigh if int(labels[int(j)]) == c_seed and int(j) != int(si)]
        cross_list = [j for j in neigh if int(labels[int(j)]) != c_seed and int(j) != int(si)]
        rng.shuffle(same_list)
        rng.shuffle(cross_list)

        while len(chosen_idx) < K and (same_list or cross_list):
            want_same = (rng.rand() < float(same_cluster_ratio))
            pick_from = same_list if (want_same and same_list) else cross_list if cross_list else same_list
            if not pick_from:
                break
            j = int(pick_from.pop())
            if j in chosen_set:
                continue
            chosen_set.add(j)
            chosen_idx.append(j)

        if len(chosen_idx) < K:
            continue

        hpos = [master_ids[j] for j in chosen_idx]
        clusters = [int(labels[j]) for j in chosen_idx]

        sig = tuple(sorted(hpos))
        if sig in used_signatures:
            continue
        used_signatures.add(sig)

        sets.append({
            "seed_i": int(si),
            "seed_hpo": h_seed,
            "k": int(K),
            "idx": chosen_idx,
            "hpos": hpos,
            "clusters": clusters,
            "n_same_cluster": int(sum(1 for c in clusters if c == c_seed)),
            "n_cross_cluster": int(sum(1 for c in clusters if c != c_seed)),
        })

    return sets


# ---------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------

def render_prompt(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out

def build_phenotype_block(hpos: List[str], h2n: Dict[str, str]) -> str:
    lines = ["PHENOTYPES:"]
    for hid in hpos:
        nm = h2n.get(hid, "")
        lines.append(f"- ID: {hid}")
        lines.append(f"  Name: {nm}")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------
# B3 generation: anchors + validation + retry (MULTI-HPO)
# ---------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])")  # conservative sentence split

def _norm(s: str) -> str:
    return " ".join(_WORD_RE.findall((s or "").lower())).strip()

def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x2 = x.strip()
        if not x2:
            continue
        if x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out

# ---- generic tokens we do NOT allow as standalone anchors (too many false hits)
_GENERIC_ONE_TOKENS: Set[str] = {
    "total", "level", "concentration", "activity", "abnormal", "increase", "increased",
    "decrease", "decreased", "circulating", "plasma", "serum", "blood", "urine", "urinary",
    "bone", "hand", "finger", "thumb", "metacarpal", "phalanx", "epiphysis", "epiphyses",
    "igg", "iga", "ige", "igg2", "immunoglobulin", "immunoglobulins", "ossification",
}

def _filter_anchor(a: str) -> Optional[str]:
    a = _norm(a)
    if not a:
        return None
    toks = a.split()
    if len(toks) == 1 and toks[0] in _GENERIC_ONE_TOKENS:
        return None
    if len(toks) >= 6:  # too long: likely over-specific and rarely appears verbatim
        # keep only first 5 tokens as a softer anchor
        a2 = " ".join(toks[:5]).strip()
        if a2 and not (len(a2.split()) == 1 and a2 in _GENERIC_ONE_TOKENS):
            return a2
        return None
    return a

# Small synonym hooks (extend anytime)
# IMPORTANT: avoid ambiguous anchors that make hypo/hyper both hit on "urate"/"uric acid".
_SYNONYM_ANCHORS: Dict[str, List[str]] = {
    "hypouricemia": [
        "hypouricemia",
        "low uric acid",
        "low serum uric acid",
        "low urate",
        "low serum urate",
    ],
    "hyperuricemia": [
        "hyperuricemia",
        "high uric acid",
        "high serum uric acid",
        "high urate",
        "high serum urate",
    ],
    "carnosinuria": ["carnosinuria", "urinary carnosine", "carnosine in the urine"],
    "methioninuria": ["methioninuria", "urinary methionine", "methionine in the urine"],
    "abnormal urine urobilinogen level": ["urobilinogen", "urine urobilinogen", "urinary urobilinogen"],
    "decreased circulating hydroxyproline concentration": ["low hydroxyproline", "reduced hydroxyproline", "hydroxyproline deficiency"],
    "abnormal circulating nucleobase concentration": ["abnormal nucleobase", "abnormal nucleobases", "purine abnormality", "pyrimidine abnormality"],
    "decreased plasma carnitine": ["low carnitine", "reduced carnitine", "carnitine deficiency"],
    "hypernatriuria": [
        "hypernatriuria",
        "increased urinary sodium",
        "elevated urinary sodium",
        "increased sodium excretion",
        "elevated urine sodium",
        "increased urine sodium",
    ],
    "hypovalinemia": ["hypovalinemia", "low valine", "low plasma valine", "reduced plasma valine"],
    "hypervalinemia": ["hypervalinemia", "high valine", "elevated plasma valine", "increased plasma valine"],
}

_DEFAULT_BLACKLIST = [
    "alkaline phosphatase",
    "acid phosphatase",
    "arylsulfatase",
    "erythematous",
    "rash",
    "macular rash",
]

def build_anchors_from_name(hpo_name: str) -> List[str]:
    """
    Build anchors that are:
      - specific enough to avoid cross-HPO inflation
      - still likely to appear in clinical-style sentences
    """
    name0 = (hpo_name or "").strip()
    if not name0:
        return []
    key = name0.lower().strip()

    # Use curated synonyms if available
    if key in _SYNONYM_ANCHORS:
        anchors = []
        for x in _SYNONYM_ANCHORS[key]:
            a = _filter_anchor(x)
            if a:
                anchors.append(a)
        return _unique_keep_order(anchors)

    # IMPORTANT: do NOT drop polarity/dynamics tokens; they help distinguish opposing phenotypes.
    drop = {
        "abnormal", "circulating", "plasma", "serum", "blood", "of", "the", "and",
        "level", "concentration", "activity",
    }
    toks = [t for t in _WORD_RE.findall(key) if t and t not in drop]

    anchors: List[str] = []
    if toks:
        anchors.append(" ".join(toks[:3]))
        anchors.append(" ".join(toks[:4]))
        anchors.extend(toks[:5])  # may include polarity tokens like hyper/hypo/increased/decreased/chronic/transient
    anchors.append(_norm(name0))

    anchors2: List[str] = []
    for a in anchors:
        aa = _filter_anchor(a)
        if aa:
            anchors2.append(aa)
    return _unique_keep_order(anchors2)

def parse_lines_plain(text: str) -> List[str]:
    """
    Robust parsing:
      1) splitlines
      2) if only one long line, split into sentences
      3) strip bullets/numbers
    """
    raw = (text or "").strip()
    if not raw:
        return []

    lines0 = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # If LLM outputs a single paragraph, sentence-split.
    if len(lines0) <= 1:
        if len(raw) >= 120:
            parts = _SENT_SPLIT_RE.split(raw)
            lines0 = [p.strip() for p in parts if p.strip()]

    lines: List[str] = []
    for ln in lines0:
        s = re.sub(r"^\s*[\-\*\d\.\)\]]+\s*", "", ln).strip()
        if s:
            lines.append(s)
    return lines

def line_hits_blacklist(line: str, blacklist: List[str]) -> bool:
    t = (line or "").lower()
    return any(b in t for b in blacklist)

def _contains_anchor(norm_line: str, anchor: str) -> bool:
    """
    Safer than raw substring: require word-boundary style containment
    after normalization (space-separated tokens).
    """
    if not anchor:
        return False
    hay = f" {norm_line} "
    nee = f" {anchor} "
    return nee in hay

def count_hits_per_phenotype(line: str, anchors_by_hpo: Dict[str, List[str]]) -> Tuple[int, List[str]]:
    t = _norm(line)
    hit_hpos: List[str] = []
    for hid, anchors in anchors_by_hpo.items():
        for a in anchors:
            if a and _contains_anchor(t, a):
                hit_hpos.append(hid)
                break
    hit_hpos = _unique_keep_order(hit_hpos)
    return (len(hit_hpos), hit_hpos)

def required_keywords_block_multi(
    anchors_by_hpo: Dict[str, List[str]],
    min_hits_per_line: int,
    *,
    must_include_hpo: str = "",
    max_show_per_hpo: int = 2,
    extra_rules: Optional[List[str]] = None,
) -> str:
    lines = [
        "REQUIRED KEYWORDS (HARD CONSTRAINT):",
        f"- Each line MUST mention at least {min_hits_per_line} phenotypes from the list below (via keywords).",
    ]
    if must_include_hpo:
        lines.append(f"- Additionally, EACH line MUST include the seed phenotype: {must_include_hpo}.")
    lines += [
        "- Do NOT introduce unrelated findings or phenotypes beyond the provided list.",
    ]
    if extra_rules:
        lines += [f"- {x}" for x in extra_rules if x.strip()]

    lines += ["", "PHENOTYPE KEYWORDS:"]
    for hid, anchors in anchors_by_hpo.items():
        show = ", ".join([a for a in anchors[:max_show_per_hpo] if a])
        lines.append(f"- {hid}: {show}".strip())
    return "\n".join(lines).strip()

def filter_and_score_lines_multi(
    lines: List[str],
    *,
    anchors_by_hpo: Dict[str, List[str]],
    min_hits_per_line: int,
    use_blacklist: bool,
    blacklist: List[str],
    must_include_seed: bool,
    seed_hpo: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns valid lines + debug stats.
    Valid means:
      - (optional) not in blacklist
      - hit_n >= min_hits_per_line
      - (optional) includes seed_hpo
    """
    valid: List[str] = []
    per_line_hits: List[Dict[str, Any]] = []
    covered: Set[str] = set()

    for ln in lines:
        if use_blacklist and line_hits_blacklist(ln, blacklist):
            continue

        hit_n, hit_hpos = count_hits_per_phenotype(ln, anchors_by_hpo)

        if must_include_seed and seed_hpo:
            if seed_hpo not in hit_hpos:
                per_line_hits.append({"line": ln, "hit_n": hit_n, "hit_hpos": hit_hpos, "seed_ok": False})
                continue

        per_line_hits.append({"line": ln, "hit_n": hit_n, "hit_hpos": hit_hpos, "seed_ok": True})

        if hit_n >= int(min_hits_per_line):
            valid.append(ln)
            for hid in hit_hpos:
                covered.add(hid)

    valid = _unique_keep_order(valid)

    covered_all = sorted(list(covered))
    debug = {
        "n_raw_lines": len(lines),
        "n_valid_lines": len(valid),
        "min_hits_per_line": int(min_hits_per_line),
        "must_include_seed": bool(must_include_seed),
        "coverage_n": len(covered_all),
        "coverage_ratio": (len(covered_all) / max(1, len(anchors_by_hpo))),
        "per_line_hits_preview": per_line_hits[:6],
        "covered_hpos_all": covered_all,
        "covered_hpos_preview": covered_all[:50],
    }
    return valid, debug

def _diversity_stats(lines: List[str]) -> Dict[str, Any]:
    """
    Lightweight diversity check to catch template collapse.
    """
    if not lines:
        return {"div_ok": False, "reason": "no_lines", "dup_start_ratio": 1.0, "near_dup_pairs": 0}

    starts = []
    norm_lines = []
    for ln in lines:
        toks = _norm(ln).split()
        starts.append(" ".join(toks[:3]) if toks else "")
        norm_lines.append(" ".join(toks))

    uniq_start = len(set([s for s in starts if s]))
    dup_start_ratio = 1.0 - (uniq_start / max(1, len(starts)))

    # near-duplicate pairs using Jaccard on token sets
    near_dup_pairs = 0
    sets = [set(x.split()) for x in norm_lines]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a or not b:
                continue
            jac = len(a & b) / max(1, len(a | b))
            if jac >= 0.92:
                near_dup_pairs += 1

    # simple policy: allow some duplication, but not collapse
    div_ok = True
    reason = "ok"
    if dup_start_ratio > 0.55:
        div_ok = False
        reason = "duplicate_starts"
    if near_dup_pairs >= max(3, len(lines) // 2):
        div_ok = False
        reason = "near_duplicates"

    return {
        "div_ok": div_ok,
        "reason": reason,
        "dup_start_ratio": float(dup_start_ratio),
        "near_dup_pairs": int(near_dup_pairs),
    }

def _missing_hpos(hpos: List[str], covered_all: List[str]) -> List[str]:
    cset = set(covered_all)
    return [h for h in hpos if h not in cset]

def _fmt_missing_block(missing: List[str], h2n: Dict[str, str], anchors_by_hpo: Dict[str, List[str]]) -> str:
    if not missing:
        return ""
    lines = ["MISSING PHENOTYPES (MUST APPEAR AT LEAST ONCE):"]
    for hid in missing:
        nm = h2n.get(hid, "")
        anchors = anchors_by_hpo.get(hid, [])[:3]
        lines.append(f"- {hid} | {nm} | keywords: {', '.join([a for a in anchors if a])}".strip())
    return "\n".join(lines).strip()

def generate_set_with_gating(
    *,
    llm: LLMClient,
    system_prompt: str,
    gen_template: str,
    mapping: Dict[str, str],
    hpos: List[str],
    seed_hpo: str,
    h2n: Dict[str, str],
    n_lines: int,
    temperature: float,
    max_tokens: int,
    max_retry: int,
    min_valid_lines: int,
    min_hits_per_line: int,
    require_full_coverage: bool,
    min_coverage_ratio: float,
    must_include_seed: bool,
    use_blacklist: bool,
    blacklist: List[str],
    required_keywords_show_per_hpo: int = 2,
) -> Dict[str, Any]:
    """
    Multi-HPO generation with:
      - per-line hit threshold
      - must-include-seed (optional)
      - global coverage: either full coverage, or min_coverage_ratio
      - retry with increasing strictness
      - attempt 3 = targeted repair for missing phenotypes
    """
    # Build anchors by HPO
    anchors_by_hpo: Dict[str, List[str]] = {}
    for hid in hpos:
        nm = h2n.get(hid, "")
        anchors_by_hpo[hid] = build_anchors_from_name(nm)

    anchors_ready = all(len(v) > 0 for v in anchors_by_hpo.values())

    best_valid: List[str] = []
    best_debug: Dict[str, Any] = {}
    last_raw = ""
    last_lines: List[str] = []

    # guard: min_coverage_ratio within [0,1]
    min_coverage_ratio = float(max(0.0, min(1.0, min_coverage_ratio)))

    # Keep a running "best missing" for targeted repair
    best_missing: List[str] = []

    for attempt in range(1, int(max_retry) + 1):
        mp = dict(mapping)
        mp["N"] = str(int(n_lines))
        mp.setdefault("EVIDENCE_BLOCK", mp.get("EVIDENCE", ""))
        mp.setdefault("PHENOTYPE_BLOCK", mp.get("PHENOTYPES", ""))

        # Backward compatible pair placeholders (first two phenotypes)
        if len(hpos) >= 1:
            mp.setdefault("HPO_A_ID", hpos[0])
            mp.setdefault("HPO_A_NAME", h2n.get(hpos[0], ""))
        if len(hpos) >= 2:
            mp.setdefault("HPO_B_ID", hpos[1])
            mp.setdefault("HPO_B_NAME", h2n.get(hpos[1], ""))

        user_gen = render_prompt(gen_template, mp)

        # Diversity nudge from attempt 2+
        extra_rules: List[str] = []
        if attempt >= 2:
            extra_rules += [
                "Use diverse sentence starters (do not reuse the same 3-word start across lines).",
                "Vary phrasing style (e.g., 'laboratory evaluation', 'serum analysis', 'testing shows', 'profile demonstrates').",
            ]

        # Attempt 2+: add required keyword block
        if anchors_ready and attempt >= 2:
            req_block = required_keywords_block_multi(
                anchors_by_hpo,
                int(min_hits_per_line),
                must_include_hpo=(seed_hpo if must_include_seed else ""),
                max_show_per_hpo=int(required_keywords_show_per_hpo),
                extra_rules=extra_rules,
            )
            user_gen = req_block + "\n\n" + user_gen

        # Attempt 3: targeted repair on missing phenotypes (if any)
        if anchors_ready and attempt >= 3:
            miss = best_missing[:]  # from previous best
            miss_block = _fmt_missing_block(miss, h2n, anchors_by_hpo)
            user_gen = (
                "TARGETED REPAIR ATTEMPT:\n"
                f"- Generate exactly {int(n_lines)} lines.\n"
                f"- Each line MUST mention at least {int(min_hits_per_line)} phenotypes (via keywords).\n"
                + (f"- Each line MUST include the seed phenotype: {seed_hpo}.\n" if must_include_seed else "")
                + "- You MUST ensure every phenotype in the MISSING list appears at least once across the lines.\n"
                "- Do NOT introduce any phenotypes beyond the provided list.\n\n"
                + (miss_block + "\n\n" if miss_block else "")
                + user_gen
            )

        try:
            raw = llm.complete_text(
                system=system_prompt,
                user=user_gen,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
        except Exception as e:
            raw = f"ERROR: {e}"

        last_raw = (raw or "").strip()[:12000]
        last_lines = parse_lines_plain(last_raw)

        valid, dbg = filter_and_score_lines_multi(
            last_lines,
            anchors_by_hpo=anchors_by_hpo,
            min_hits_per_line=int(min_hits_per_line),
            use_blacklist=bool(use_blacklist),
            blacklist=blacklist,
            must_include_seed=bool(must_include_seed),
            seed_hpo=seed_hpo,
        )

        # ---- Coverage from FULL list (not preview)
        covered_all = dbg.get("covered_hpos_all", [])
        if not isinstance(covered_all, list):
            covered_all = []
        coverage_ratio = float(len(set(covered_all)) / max(1, len(hpos)))

        coverage_ok = True
        if bool(require_full_coverage):
            coverage_ok = (len(set(covered_all)) == len(hpos))
        else:
            coverage_ok = (coverage_ratio >= min_coverage_ratio)

        # ---- Diversity stats on valid lines (not raw)
        div = _diversity_stats(valid)
        div_ok = bool(div.get("div_ok", True))

        # compute missing list (for next repair attempt)
        missing = _missing_hpos(hpos, covered_all)
        if (not best_missing) or (len(missing) < len(best_missing)):
            best_missing = missing

        dbg.update({
            "anchors_ready": anchors_ready,
            "attempt": attempt,
            "seed_hpo": seed_hpo,
            "n_hpos": len(hpos),
            "coverage_ratio_valid": coverage_ratio,
            "min_coverage_ratio": float(min_coverage_ratio),
            "require_full_coverage": bool(require_full_coverage),
            "coverage_ok": bool(coverage_ok),
            "missing_hpos": missing,
            "diversity": div,
        })

        # Keep best attempt by (valid_count, coverage_ratio, diversity)
        better = False
        if len(valid) > len(best_valid):
            better = True
        elif len(valid) == len(best_valid):
            prev_cov = float(best_debug.get("coverage_ratio_valid", -1.0)) if best_debug else -1.0
            if coverage_ratio > prev_cov:
                better = True
            elif abs(coverage_ratio - prev_cov) < 1e-9:
                prev_div_ok = bool((best_debug.get("diversity") or {}).get("div_ok", True)) if best_debug else True
                if div_ok and (not prev_div_ok):
                    better = True

        if better:
            best_valid = valid
            best_debug = dbg

        # Accept if meets validity + coverage; diversity is a soft constraint
        # If diversity is bad but everything else ok and no retries left, still accept.
        if len(valid) >= int(min_valid_lines) and coverage_ok:
            return {
                "ok": True,
                "attempts": attempt,
                "raw_text": last_raw,
                "lines_raw": last_lines,
                "lines_valid": valid[:int(n_lines)],
                "anchors_by_hpo": {k: v[:10] for k, v in anchors_by_hpo.items()},
                "debug": dbg,
            }

    return {
        "ok": False,
        "attempts": int(max_retry),
        "raw_text": last_raw,
        "lines_raw": last_lines,
        "lines_valid": best_valid[:int(n_lines)],
        "anchors_by_hpo": {k: v[:10] for k, v in anchors_by_hpo.items()},
        "debug": best_debug if best_debug else {
            "anchors_ready": anchors_ready,
            "n_raw_lines": len(last_lines),
            "n_valid_lines_best": len(best_valid),
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser("b3_test_driver (PLAN A+++: set-level corpus)")

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--prompts_dir", required=True)
    ap.add_argument("--recluster_dir", required=True)
    ap.add_argument("--def_source_jsonl", required=True)

    ap.add_argument("--evidence_pool_jsonl", required=True)
    ap.add_argument("--case_chunks_jsonl", default="", help="optional; improves language context (test)")

    # set sampling
    ap.add_argument("--n_sets", type=int, default=24)
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=7)
    ap.add_argument("--per_seed_candidates", type=int, default=40, help="how many knn candidates to consider per seed")
    ap.add_argument("--same_cluster_ratio", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)

    # evidence in prompt
    ap.add_argument("--max_evidence_items_per_hpo", type=int, default=1)
    ap.add_argument("--max_evidence_chars", type=int, default=1400)

    # llm
    ap.add_argument("--llm_api_key_env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--llm_base_url", default="https://api.deepseek.com")
    ap.add_argument("--llm_model", default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens_gen", type=int, default=560)

    # prompt filenames
    ap.add_argument("--system_prompt_name", default="System_Prompt.txt")
    ap.add_argument("--generate_prompt_name", default="generate.txt")

    # generation control
    ap.add_argument("--n_lines_gen", type=int, default=10)
    ap.add_argument("--gen_max_retry", type=int, default=3)
    ap.add_argument("--min_valid_lines", type=int, default=4)

    # ---- key: revised defaults for set realism
    ap.add_argument("--min_hits_per_line", type=int, default=2, help="per-line phenotype hits threshold")
    ap.add_argument("--must_include_seed", type=int, default=1, help="1=each valid line must include seed phenotype")
    ap.add_argument("--require_full_coverage", type=int, default=0, help="1=all phenotypes must appear at least once across valid lines")
    ap.add_argument("--min_coverage_ratio", type=float, default=0.7, help="if not full coverage, require >= this ratio of phenotypes covered across valid lines")

    ap.add_argument("--required_keywords_show_per_hpo", type=int, default=2, help="how many anchors to show per phenotype in REQUIRED KEYWORDS block")

    ap.add_argument("--use_blacklist", type=int, default=1)
    ap.add_argument("--blacklist_extra", default="", help="comma-separated extra blacklist phrases")

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # --------- load prompts ---------
    sys_p = os.path.join(args.prompts_dir, args.system_prompt_name)
    gen_p = os.path.join(args.prompts_dir, args.generate_prompt_name)

    system_prompt = read_text(sys_p).strip()
    gen_template = read_text(gen_p).strip()

    print("[PATH] out_dir        =", args.out_dir)
    print("[PATH] prompts_dir    =", args.prompts_dir)
    print("[PATH] recluster_dir  =", args.recluster_dir)
    print("[PATH] def_source     =", args.def_source_jsonl)
    print("[PATH] evidence_pool  =", args.evidence_pool_jsonl)
    if args.case_chunks_jsonl:
        print("[PATH] case_chunks    =", args.case_chunks_jsonl)

    # --------- load recluster artifacts ---------
    master_ids = load_master_ids(args.recluster_dir)
    knn_idx = load_knn_round2(args.recluster_dir)
    labels = load_labels_round2(args.recluster_dir)
    V = load_v1_smooth(args.recluster_dir)

    print(f"[LOAD] master_ids: {len(master_ids)}")
    print(f"[LOAD] knn_idx   : {knn_idx.shape}")
    print(f"[LOAD] labels    : {labels.shape}")
    print(f"[LOAD] V1_smooth : {V.shape if V is not None else '(missing, ok)'}")

    # --------- names ---------
    h2n = load_hpo_names(args.def_source_jsonl)
    print(f"[LOAD] hpo names : {len(h2n)}")

    # --------- evidence indices (test-mode) ---------
    print("[INDEX] building pubmed_by_hpo from evidence_pool.jsonl ...")
    pubmed_by_hpo = build_pubmed_pool_index_by_hpo(
        args.evidence_pool_jsonl,
        max_items_per_hpo=max(0, int(args.max_evidence_items_per_hpo)),
    )
    print(f"[INDEX] pubmed_by_hpo: {len(pubmed_by_hpo)} HPOs with <= {args.max_evidence_items_per_hpo} items")

    case_chunks = None
    if args.case_chunks_jsonl and os.path.isfile(args.case_chunks_jsonl):
        print("[INDEX] loading case_chunks (test-mode cache) ...")
        case_chunks = build_case_chunks_cache(args.case_chunks_jsonl, max_keep=200000)
        print(f"[INDEX] case_chunks cached: {len(case_chunks)}")

    # --------- sample sets ---------
    sets = sample_sets_from_knn(
        master_ids, knn_idx, labels,
        n_sets=int(args.n_sets),
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        per_seed_candidates=int(args.per_seed_candidates),
        same_cluster_ratio=float(args.same_cluster_ratio),
        seed=int(args.seed),
    )
    print(f"[SAMPLE] sampled sets: {len(sets)}")

    sampled_path = os.path.join(args.out_dir, "sets_sampled.jsonl")
    write_jsonl(sampled_path, sets)

    # --------- init LLM client ---------
    api_key = os.environ.get(args.llm_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key env: {args.llm_api_key_env}")

    llm = LLMClient(
        api_key=api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout=float(args.timeout),
    )

    # --------- blacklist ---------
    blacklist = list(_DEFAULT_BLACKLIST)
    if args.blacklist_extra.strip():
        for x in args.blacklist_extra.split(","):
            x = x.strip().lower()
            if x:
                blacklist.append(x)
    use_blacklist = bool(int(args.use_blacklist))

    # --------- save run config ---------
    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": now_tag(),
        "paths": {
            "out_dir": args.out_dir,
            "prompts_dir": args.prompts_dir,
            "recluster_dir": args.recluster_dir,
            "def_source_jsonl": args.def_source_jsonl,
            "evidence_pool_jsonl": args.evidence_pool_jsonl,
            "case_chunks_jsonl": args.case_chunks_jsonl,
        },
        "sampling": {
            "plan": "A+++ (set-level corpus item)",
            "n_sets": int(args.n_sets),
            "k_min": int(args.k_min),
            "k_max": int(args.k_max),
            "per_seed_candidates": int(args.per_seed_candidates),
            "same_cluster_ratio": float(args.same_cluster_ratio),
            "seed": int(args.seed),
            "knn_source": "knn_round2_idx.npy",
            "labels_source": "labels_round2.npy",
        },
        "prompt_evidence": {
            "max_evidence_items_per_hpo": int(args.max_evidence_items_per_hpo),
            "max_evidence_chars": int(args.max_evidence_chars),
            "use_case_chunks": bool(case_chunks is not None),
        },
        "generation_control": {
            "n_lines_gen": int(args.n_lines_gen),
            "min_hits_per_line": int(args.min_hits_per_line),
            "must_include_seed": bool(int(args.must_include_seed)),
            "require_full_coverage": bool(int(args.require_full_coverage)),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "gen_max_retry": int(args.gen_max_retry),
            "min_valid_lines": int(args.min_valid_lines),
            "use_blacklist": use_blacklist,
            "blacklist_size": len(blacklist),
            "required_keywords_show_per_hpo": int(args.required_keywords_show_per_hpo),
            "parser": "splitlines + sentence-split fallback",
            "coverage_fix": "covered_hpos_all used for coverage_ok",
            "attempt3": "targeted missing-phenotype repair",
            "anchor_filter": "drop overly-generic one-token anchors; avoid ambiguous hypo/hyper anchors",
            "diversity_check": "duplicate starters + near-duplicate Jaccard",
        },
        "llm": {
            "base_url": args.llm_base_url,
            "model": args.llm_model,
            "timeout": float(args.timeout),
            "temperature": float(args.temperature),
            "max_tokens_gen": int(args.max_tokens_gen),
            "plain_text_only": True,
        },
        "notes": [
            "TEST MODE: evidence is lightweight (PubMed by HPO + optional deterministic PMC chunk).",
            "Set-level gating: per-line hits + seed-include + global coverage ratio (or full coverage).",
            "Attempt 3 is a targeted repair using missing phenotypes extracted from previous best attempt.",
        ],
    }
    cfg_path = os.path.join(args.out_dir, "B3_run_config.json")
    write_json(cfg_path, run_config)

    # --------- generate corpus for each set ---------
    corpus_out: List[Dict[str, Any]] = []

    pbar = tqdm(sets, desc="B3 set-generate", unit="set")
    for srec in pbar:
        hpos: List[str] = list(srec["hpos"])
        seed_hpo: str = str(srec.get("seed_hpo", hpos[0] if hpos else ""))

        names = [h2n.get(h, "") for h in hpos]

        # evidence block
        ev_text = collect_evidence_for_set(
            hpos,
            h2n=h2n,
            pubmed_by_hpo=pubmed_by_hpo,
            case_chunks=case_chunks,
            max_items_per_hpo=int(args.max_evidence_items_per_hpo),
            max_chars=int(args.max_evidence_chars),
        )

        phen_block = build_phenotype_block(hpos, h2n)

        # optional: quick internal similarity signal using V1 (seed vs others avg)
        sim_seed_mean = None
        if V is not None:
            try:
                seed_i = int(srec["seed_i"])
                idxs = [int(x) for x in srec["idx"]]
                sims = [float(np.dot(V[seed_i], V[j])) for j in idxs if j != seed_i]
                sim_seed_mean = float(sum(sims) / max(1, len(sims)))
            except Exception:
                sim_seed_mean = None

        base_mapping = {
            "EVIDENCE": ev_text,
            "EVIDENCE_BLOCK": ev_text,
            "PHENOTYPES": phen_block,
            "PHENOTYPE_BLOCK": phen_block,
            "N": str(int(args.n_lines_gen)),
            # backward-compatible "pair" placeholders (first two)
            "HPO_A_ID": hpos[0] if len(hpos) > 0 else "",
            "HPO_A_NAME": names[0] if len(names) > 0 else "",
            "HPO_B_ID": hpos[1] if len(hpos) > 1 else "",
            "HPO_B_NAME": names[1] if len(names) > 1 else "",
            "SIM": "" if sim_seed_mean is None else f"{sim_seed_mean:.4f}",
        }

        gen_pack = generate_set_with_gating(
            llm=llm,
            system_prompt=system_prompt,
            gen_template=gen_template,
            mapping=base_mapping,
            hpos=hpos,
            seed_hpo=seed_hpo,
            h2n=h2n,
            n_lines=int(args.n_lines_gen),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens_gen),
            max_retry=int(args.gen_max_retry),
            min_valid_lines=int(args.min_valid_lines),
            min_hits_per_line=int(args.min_hits_per_line),
            require_full_coverage=bool(int(args.require_full_coverage)),
            min_coverage_ratio=float(args.min_coverage_ratio),
            must_include_seed=bool(int(args.must_include_seed)),
            use_blacklist=use_blacklist,
            blacklist=blacklist,
            required_keywords_show_per_hpo=int(args.required_keywords_show_per_hpo),
        )

        corpus_out.append({
            "item_type": "phenotype_set",
            "seed_i": srec["seed_i"],
            "seed_hpo": seed_hpo,
            "k": srec["k"],
            "hpos": hpos,
            "names": names,
            "clusters": srec["clusters"],
            "n_same_cluster": srec["n_same_cluster"],
            "n_cross_cluster": srec["n_cross_cluster"],
            "sim_seed_mean": sim_seed_mean,
            "gen_ok": bool(gen_pack["ok"]),
            "gen_attempts": int(gen_pack["attempts"]),
            "min_hits_per_line": int(args.min_hits_per_line),
            "must_include_seed": bool(int(args.must_include_seed)),
            "require_full_coverage": bool(int(args.require_full_coverage)),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "template_lines": gen_pack["lines_valid"],
            "template_raw": (gen_pack["raw_text"] or "").strip()[:8000],
            "anchors_by_hpo": gen_pack["anchors_by_hpo"],
            "debug": gen_pack["debug"],
        })

    out_c = os.path.join(args.out_dir, "corpus_generated.jsonl")
    write_jsonl(out_c, corpus_out)

    # --------- summary ---------
    n_ok = sum(1 for x in corpus_out if x.get("gen_ok") is True)
    n_fail = sum(1 for x in corpus_out if x.get("gen_ok") is False)
    avg_attempts = float(sum(int(x.get("gen_attempts", 0)) for x in corpus_out) / max(1, len(corpus_out)))

    # coverage stats (from debug; prefer coverage_ratio_valid)
    covs = []
    for x in corpus_out:
        dbg = x.get("debug", {}) or {}
        try:
            covs.append(float(dbg.get("coverage_ratio_valid", dbg.get("coverage_ratio", 0.0))))
        except Exception:
            pass
    cov_mean = float(sum(covs) / max(1, len(covs))) if covs else 0.0

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_sets": len(sets),
        "generation_counts": {
            "items_total": len(corpus_out),
            "gen_ok": n_ok,
            "gen_fail": n_fail,
            "avg_attempts": avg_attempts,
        },
        "coverage_stats": {
            "mean_coverage_ratio_valid": cov_mean,
            "n_items_with_cov": len(covs),
        },
        "paths": {
            "config": cfg_path,
            "sets_sampled": sampled_path,
            "corpus_generated": out_c,
        }
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)

    print("\n[DONE] B3 set-level outputs:")
    print("  -", cfg_path)
    print("  -", sampled_path)
    print("  -", out_c)
    print("  -", os.path.join(args.out_dir, "summary.json"))
    print("[GEN]", json.dumps(summary["generation_counts"], ensure_ascii=False, indent=2))
    print("[COV]", json.dumps(summary["coverage_stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
