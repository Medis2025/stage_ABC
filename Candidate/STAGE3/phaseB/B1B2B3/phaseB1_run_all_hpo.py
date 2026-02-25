#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
phaseB1_run_all_hpo.py
(PUBMED-ABS MAIN + PMC OPTIONAL + FULL RUN + RESUME + CONFIG-ONCE + TQDM + THREADPOOL + RETRY)

✅ PubMed (PMID + abstract) as MAIN reference (high coverage).
✅ PMC (PMCID + chunk candidates) optional supplement (strong evidence).
✅ Resume key: (doc_source, hpo_id, doc_id, chunk_key)
✅ Adds B1 output sanitation:
   - parse lines, de-dup, clamp max lines (NO min lines)
   - allow NONE as a valid "no-evidence" output (NOT LLM failure)
   - write NONE outcomes to failures.jsonl as stage=no_evidence_none
✅ Adds structured output_lines (preferred for downstream B2/B3)
✅ Adds prompt support for optional {HPO_LLM_DEF} / {{HPO_LLM_DEF}} placeholder

Doc modes
---------
--doc_mode pubmed_only : only PubMed abstract jobs
--doc_mode pmc_only    : only PMC candidate jobs
--doc_mode hybrid      : PubMed jobs + optional PMC jobs (recommended)

PhaseA artifacts
----------------
Required for pubmed_only / hybrid:
  <phaseA_run_dir>/pubmed/hpo_to_pmids.jsonl
  <phaseA_run_dir>/pubmed/pmid_to_abstract.jsonl

Optional for hybrid (or required for pmc_only):
  <phaseA_run_dir>/pubmed/hpo_to_pmcids.jsonl
  <phaseA_run_dir>/phaseA/merged/mentions_candidates.jsonl

Usage example (hybrid)
----------------------
python3 phaseB1_run_all_hpo.py \
  --phaseA_run_dir "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/out_phaseA/20260126_164618" \
  --prompt_txt "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/prompts/B1.txt" \
  --out_base "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/out" \
  --hpo_json "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json" \
  --doc_mode hybrid \
  --pmids_per_hpo 5 \
  --max_abs_chars 1600 \
  --pmcids_per_hpo 2 \
  --cands_per_pmcid 1 \
  --max_candidates_per_pmcid 5 \
  --workers 8 \
  --retry_k 3 \
  --temperature 0.2 \
  --max_tokens 256 \
  --b1_max_lines 8 \
  --none_token "NONE"

Env
---
export DEEPSEEK_API_KEY="..."
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import random
from typing import Any, Dict, List, Optional, Iterable, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# =============================================================================
# IO helpers (append + atomic)
# =============================================================================

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def compact_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def atomic_write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def read_jsonl_iter(path: str) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue

def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(os.path.dirname(path))
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def file_exists_nonempty(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


# =============================================================================
# Prompt handling
# =============================================================================

def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fill_prompt(
    tpl: str,
    hpo_id: str,
    hpo_name: str,
    hpo_def: str,
    hpo_llm_def: str,
    context: str
) -> str:
    """
    Supports placeholders:
      {HPO_ID}, {HPO_NAME}, {HPO_DEF}, {HPO_LLM_DEF}, {CONTEXT}
      and double-brace variants {{HPO_DEF}} etc.

    If placeholders not present, appends an INPUT block at end.
    """
    rep_pairs = {
        "{HPO_ID}": hpo_id,
        "{HPO_NAME}": hpo_name,
        "{HPO_DEF}": hpo_def,
        "{HPO_LLM_DEF}": hpo_llm_def,
        "{CONTEXT}": context,
        "{{HPO_ID}}": hpo_id,
        "{{HPO_NAME}}": hpo_name,
        "{{HPO_DEF}}": hpo_def,
        "{{HPO_LLM_DEF}}": hpo_llm_def,
        "{{CONTEXT}}": context,
    }

    out = tpl
    replaced_any = False
    for k, v in rep_pairs.items():
        if k in out:
            out = out.replace(k, v)
            replaced_any = True

    if not replaced_any:
        out = (
            tpl.rstrip()
            + "\n\nINPUT:\n"
            + f"[HPO_ID] {hpo_id}\n"
            + f"[HPO_NAME] {hpo_name}\n"
            + f"[HPO_DEF] {hpo_def}\n"
            + f"[HPO_LLM_DEF] {hpo_llm_def}\n\n"
            + "[CONTEXT]\n"
            + context
            + "\n\nOUTPUT:\n"
        )
    return out


# =============================================================================
# HPO JSON: primary source (Name + Def integration)
# =============================================================================

def _first_str(v: Any) -> str:
    if isinstance(v, list) and v:
        return safe_str(v[0]).strip()
    if isinstance(v, str):
        return v.strip()
    return ""

def join_text_list(v: Any, max_items: int = 4) -> str:
    if isinstance(v, list):
        items = [safe_str(x).strip() for x in v if safe_str(x).strip()]
        if not items:
            return ""
        return " ".join(items[:max_items]).strip()
    if isinstance(v, str):
        return v.strip()
    return ""

def build_hpo_def(obj: Dict[str, Any], max_len: int = 900) -> str:
    """
    Compact definition string for the prompt.
    Prefer llm_def / Def, then llm_add_def, then Comment, then Synonym.
    """
    parts: List[str] = []

    llm_def = _first_str(obj.get("llm_def"))
    if llm_def:
        parts.append(llm_def)

    d = join_text_list(obj.get("Def"), max_items=2)
    if d and d not in " ".join(parts):
        parts.append(d)

    llm_add = _first_str(obj.get("llm_add_def"))
    if llm_add:
        parts.append(llm_add)

    cm = join_text_list(obj.get("Comment"), max_items=2)
    if cm:
        parts.append(cm)

    syn = join_text_list(obj.get("Synonym"), max_items=4)
    if syn:
        parts.append("Synonyms: " + syn)

    out = compact_ws(" ".join(parts))
    if max_len and len(out) > max_len:
        out = out[:max_len].rstrip() + " ..."
    return out

def build_hpo_llm_def(obj: Dict[str, Any], max_len: int = 700) -> str:
    """
    Optional extra guidance for prompt slot {HPO_LLM_DEF}.
    We keep it small and conservative:
      - llm_def (if exists)
      - llm_add_def (if exists)
    """
    parts: List[str] = []
    llm_def = _first_str(obj.get("llm_def"))
    if llm_def:
        parts.append(llm_def)
    llm_add = _first_str(obj.get("llm_add_def"))
    if llm_add and llm_add not in " ".join(parts):
        parts.append(llm_add)
    out = compact_ws(" ".join(parts))
    if max_len and len(out) > max_len:
        out = out[:max_len].rstrip() + " ..."
    return out

def load_hpo_json_primary(
    hpo_json_path: str,
    exclude_root: bool = True,
    max_def_chars: int = 900,
    max_llmdef_chars: int = 700,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Returns:
      - hpo_meta: hpo_id -> {"name":..., "def":..., "llm_def":...}
      - pool: list of hpo_ids
    """
    if not hpo_json_path or not os.path.exists(hpo_json_path):
        raise FileNotFoundError(f"--hpo_json not found: {hpo_json_path}")

    with open(hpo_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("hpo_json must be a dict keyed by HPO IDs.")

    hpo_meta: Dict[str, Dict[str, str]] = {}
    pool: List[str] = []

    for hid, obj in data.items():
        hid = safe_str(hid).strip()
        if not hid.startswith("HP:"):
            continue
        if exclude_root and hid == "HP:0000001":
            continue
        if not isinstance(obj, dict):
            continue

        name = _first_str(obj.get("Name")) or _first_str(obj.get("Synonym"))
        hdef = build_hpo_def(obj, max_len=int(max_def_chars) if max_def_chars else 900)
        hllm = build_hpo_llm_def(obj, max_len=int(max_llmdef_chars) if max_llmdef_chars else 700)

        hpo_meta[hid] = {"name": name, "def": hdef, "llm_def": hllm}
        pool.append(hid)

    if not pool:
        raise RuntimeError("No usable HPO IDs found in hpo_json.")

    return hpo_meta, pool


# =============================================================================
# PubMed mappings
# =============================================================================

def load_hpo_to_pmids(path: str) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    for r in read_jsonl_iter(path):
        hpo_id = (r.get("hpo_id") or "").strip()
        pmids = r.get("pmids") or []
        if not hpo_id or not isinstance(pmids, list):
            continue
        seen: Set[str] = set()
        out: List[str] = []
        for pmid in pmids:
            p = safe_str(pmid).strip()
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
        if out:
            mp[hpo_id] = out
    return mp

def load_pmid_to_abstract_filtered(path: str, target_pmids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """
    Keep only pmids we will touch (memory friendly).
    Accepts rows that include "pmid"/"PMID" and "abstract"/"title" keys.
    """
    out: Dict[str, Dict[str, Any]] = {}
    with tqdm(total=None, desc="Index pmid_to_abstract (filtered)", unit="rows") as pbar:
        for r in read_jsonl_iter(path):
            pbar.update(1)
            pmid = safe_str(r.get("pmid") or r.get("PMID") or "").strip()
            if not pmid or pmid not in target_pmids:
                continue
            out[pmid] = r
    return out

def format_abstract_block(r: Dict[str, Any], max_chars: int = 1600) -> str:
    title = compact_ws(r.get("title") or r.get("Title") or "")
    abst = compact_ws(r.get("abstract") or r.get("Abstract") or r.get("abs") or "")
    journal = compact_ws(r.get("journal") or r.get("Journal") or "")
    year = compact_ws(r.get("year") or r.get("Year") or "")
    authors = compact_ws(r.get("authors") or r.get("Authors") or "")

    meta_parts: List[str] = []
    if journal:
        meta_parts.append(f"JOURNAL: {journal}")
    if year:
        meta_parts.append(f"YEAR: {year}")
    if authors:
        meta_parts.append(f"AUTHORS: {authors}")
    meta = " | ".join(meta_parts)

    lines: List[str] = []
    if title:
        lines.append(f"TITLE: {title}")
    if meta:
        lines.append(meta)
    if abst:
        lines.append("ABSTRACT:")
        lines.append(abst)

    text = "\n".join(lines).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


# =============================================================================
# PMC mapping + Candidate index (optional)
# =============================================================================

def load_hpo_to_pmcids(hpo_to_pmcids_path: str) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    for r in read_jsonl_iter(hpo_to_pmcids_path):
        hpo_id = (r.get("hpo_id") or "").strip()
        pmcids = r.get("pmcids") or []
        if not hpo_id or not isinstance(pmcids, list):
            continue
        seen: Set[str] = set()
        out: List[str] = []
        for pmcid in pmcids:
            p = safe_str(pmcid).strip()
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
        if out:
            mp[hpo_id] = out
    return mp

def reservoir_keep(bucket: List[Dict[str, Any]],
                   row: Dict[str, Any],
                   max_k: int,
                   seen_n: int,
                   rng: random.Random) -> None:
    if len(bucket) < max_k:
        bucket.append(row)
    else:
        j = rng.randint(1, seen_n)  # 1..seen_n
        if j <= max_k:
            bucket[j - 1] = row

def build_pmcid_to_candidates_index_filtered(
    merged_candidates_path: str,
    target_pmcids: Set[str],
    max_per_pmcid: int,
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    max_per_pmcid = int(max_per_pmcid)
    if max_per_pmcid <= 0:
        raise ValueError("--max_candidates_per_pmcid must be > 0")

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}

    with tqdm(total=None, desc="Index merged candidates (filtered)", unit="rows") as pbar:
        for row in read_jsonl_iter(merged_candidates_path):
            pbar.update(1)
            pmcid = safe_str(row.get("pmcid") or "").strip()
            if not pmcid or pmcid not in target_pmcids:
                continue

            counts[pmcid] = counts.get(pmcid, 0) + 1
            n = counts[pmcid]

            if pmcid not in buckets:
                buckets[pmcid] = []
            reservoir_keep(buckets[pmcid], row, max_per_pmcid, n, rng)

    return buckets


# =============================================================================
# Context build
# =============================================================================

def build_context_from_candidate(c: Dict[str, Any], max_ctx_chars: int = 1800) -> str:
    ctx = compact_ws(c.get("context") or "")
    if max_ctx_chars and len(ctx) > max_ctx_chars:
        ctx = ctx[:max_ctx_chars].rstrip() + " ..."

    surface = compact_ws(c.get("surface") or "")
    unit = compact_ws(c.get("unit") or "")
    labels = c.get("label_options") or []

    meta_lines: List[str] = []
    if surface:
        meta_lines.append(f"SURFACE: {surface}")
    if unit:
        meta_lines.append(f"UNIT: {unit}")
    if isinstance(labels, list) and labels:
        meta_lines.append("LABEL_OPTIONS: " + ", ".join([safe_str(x) for x in labels[:12]]))

    meta = "\n".join(meta_lines)
    if meta:
        return (meta + "\n\nCONTEXT:\n" + ctx).strip()
    return ctx


# =============================================================================
# B1 output sanitation (CRITICAL for downstream B2/B3)
# =============================================================================

def parse_b1_lines(text: str) -> List[str]:
    """
    Parse plain-text output into clean phrase lines:
      - strip whitespace
      - drop empty lines
      - dedup (case-insensitive; collapse whitespace)
    """
    if text is None:
        return []
    raw = [ln.strip() for ln in str(text).splitlines()]
    raw = [ln for ln in raw if ln]
    seen: Set[str] = set()
    out: List[str] = []
    for ln in raw:
        key = re.sub(r"\s+", " ", ln).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(re.sub(r"\s+", " ", ln).strip())
    return out

def is_none_output(lines: List[str], none_token: str = "NONE") -> bool:
    return len(lines) == 1 and lines[0].strip().upper() == (none_token or "NONE").strip().upper()

def clamp_phrases(lines: List[str], max_lines: int = 8) -> List[str]:
    """
    Enforce a maximum number of lines only.
    DO NOT enforce a minimum (allow 1 line or NONE).
    """
    if max_lines and len(lines) > max_lines:
        return lines[:max_lines]
    return lines

def filter_b1_lines_by_context(lines: List[str], context: str) -> List[str]:
    """
    Keep only those B1 lines that are anchored in the given context (PubMed abstract).
    Minimal, conservative rule:
      - normalize whitespace + lowercase
      - require the full phrase (or its simple singular form) to appear as a substring
    This blocks pure def-driven hallucinations that never appear in the document.
    """
    ctx = (context or "").lower()
    if not ctx or not lines:
        return []

    out: List[str] = []
    for ln in lines:
        norm = re.sub(r"\s+", " ", ln).strip()
        if not norm:
            continue
        norm_l = norm.lower()

        # exact phrase match
        if norm_l in ctx:
            out.append(norm)
            continue

        # simple plural-stripping fallback (trailing 's')
        if norm_l.endswith("s") and norm_l[:-1] in ctx:
            out.append(norm)
            continue

        # if neither matches, drop this line
    return out


# =============================================================================
# Resume handling (doc_source aware)
# =============================================================================

def make_item_key(doc_source: str, hpo_id: str, doc_id: str, chunk_key: str) -> str:
    return f"{doc_source}\t{hpo_id}\t{doc_id}\t{chunk_key}"

def load_done_keys(outputs_path: str) -> Set[str]:
    """
    Resume key:
      (doc_source, hpo_id, doc_id, chunk_key)

    Backward compatibility:
      if doc_source missing, treat as "pmc"
      if doc_id missing, use pmcid/pmid
    """
    done: Set[str] = set()
    if not file_exists_nonempty(outputs_path):
        return done

    for r in read_jsonl_iter(outputs_path):
        hid = safe_str(r.get("hpo_id") or "").strip()
        chunk_key = safe_str(r.get("chunk_key") or "").strip()
        doc_source = safe_str(r.get("doc_source") or "").strip() or "pmc"

        doc_id = safe_str(r.get("doc_id") or "").strip()
        if not doc_id:
            if doc_source == "pubmed":
                doc_id = safe_str(r.get("pmid") or "").strip()
            else:
                doc_id = safe_str(r.get("pmcid") or "").strip()

        if hid and doc_source and doc_id and chunk_key:
            done.add(make_item_key(doc_source, hid, doc_id, chunk_key))

    return done


# =============================================================================
# LLM wrapper + retry (thread-safe)
# =============================================================================

def call_llm_plain_text(llm: Any, system: str, user: str,
                        temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
    if hasattr(llm, "complete_text"):
        kw = {"system": system, "user": user, "temperature": temperature}
        if max_tokens is not None:
            kw["max_tokens"] = int(max_tokens)
        return str(llm.complete_text(**kw))

    if hasattr(llm, "complete"):
        kw = {"system": system, "user": user, "temperature": temperature}
        if max_tokens is not None:
            kw["max_tokens"] = int(max_tokens)
        return str(llm.complete(**kw))

    if hasattr(llm, "chat"):
        prompt = (system.strip() + "\n\n" + user.strip()).strip()
        try:
            return str(llm.chat(prompt, temperature=temperature, max_tokens=max_tokens))
        except TypeError:
            return str(llm.chat(prompt))

    raise AttributeError("LLMClient must provide one of: complete_text / complete / chat")

def run_one_item_with_retry(
    llm: Any,
    system_msg: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retry_k: int,
    backoff_sec: float,
) -> Tuple[bool, str]:
    last_err = ""
    for attempt in range(1, int(retry_k) + 1):
        try:
            text = call_llm_plain_text(
                llm,
                system=system_msg,
                user=prompt,
                temperature=float(temperature),
                max_tokens=int(max_tokens) if max_tokens else None,
            )
            out = (text or "").strip()
            if out:
                return True, out
            last_err = "empty_output"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        if attempt < int(retry_k):
            time.sleep(float(backoff_sec) * attempt)

    return False, last_err or "unknown_error"


# =============================================================================
# Job building (PubMed as main)
# =============================================================================

def iter_jobs_pubmed(
    hpo_ids: List[str],
    hpo_meta: Dict[str, Dict[str, str]],
    hpo_to_pmids: Dict[str, List[str]],
    pmid_to_abs: Dict[str, Dict[str, Any]],
    pmids_per_hpo: int,
    max_abs_chars: int,
    rng: random.Random,
    done_keys: Set[str],
):
    for h_i, hpo_id in enumerate(hpo_ids, start=1):
        pmids = hpo_to_pmids.get(hpo_id, [])
        if not pmids:
            yield {"kind": "fail", "row": {"stage": "no_pmids_for_hpo", "hpo_id": hpo_id, "hpo_name": hpo_meta.get(hpo_id, {}).get("name", "")}}
            continue

        k = min(int(pmids_per_hpo), len(pmids))
        chosen = rng.sample(pmids, k) if len(pmids) >= k else pmids

        for p_i, pmid in enumerate(chosen, start=1):
            r = pmid_to_abs.get(pmid)
            if not r:
                yield {"kind": "fail", "row": {"stage": "missing_pmid_abstract", "hpo_id": hpo_id, "hpo_name": hpo_meta.get(hpo_id, {}).get("name", ""), "pmid": pmid}}
                continue

            chunk_key = f"pmid:{pmid}"
            key = make_item_key("pubmed", hpo_id, pmid, chunk_key)
            if key in done_keys:
                continue

            yield {
                "kind": "job",
                "doc_source": "pubmed",
                "doc_id": pmid,
                "pmid": pmid,
                "hpo_id": hpo_id,
                "abs_row": r,
                "context_text": format_abstract_block(r, max_chars=int(max_abs_chars)),
                "trace": {"hpo_idx": h_i, "pmid_idx": p_i},
                "chunk_key": chunk_key,
            }

def iter_jobs_pmc_optional(
    hpo_ids: List[str],
    hpo_meta: Dict[str, Dict[str, str]],
    hpo_to_pmcids: Dict[str, List[str]],
    pmcid_to_cands: Dict[str, List[Dict[str, Any]]],
    pmcids_per_hpo: int,
    cands_per_pmcid: int,
    rng: random.Random,
    done_keys: Set[str],
):
    for h_i, hpo_id in enumerate(hpo_ids, start=1):
        pmcids = hpo_to_pmcids.get(hpo_id, [])
        if not pmcids:
            yield {"kind": "fail", "row": {"stage": "no_pmcids_for_hpo", "hpo_id": hpo_id, "hpo_name": hpo_meta.get(hpo_id, {}).get("name", "")}}
            continue

        k_p = min(int(pmcids_per_hpo), len(pmcids))
        chosen_pmcids = rng.sample(pmcids, k_p) if len(pmcids) >= k_p else pmcids

        for p_i, pmcid in enumerate(chosen_pmcids, start=1):
            cand_list = pmcid_to_cands.get(pmcid, [])
            if not cand_list:
                yield {"kind": "fail", "row": {"stage": "no_candidates_for_pmcid", "hpo_id": hpo_id, "hpo_name": hpo_meta.get(hpo_id, {}).get("name", ""), "pmcid": pmcid}}
                continue

            k_c = min(int(cands_per_pmcid), len(cand_list))
            chosen_cands = rng.sample(cand_list, k_c) if len(cand_list) >= k_c else cand_list

            for c_i, cand in enumerate(chosen_cands, start=1):
                chunk_key = safe_str(cand.get("chunk_key") or "").strip()
                if not chunk_key:
                    yield {"kind": "fail", "row": {"stage": "bad_candidate", "hpo_id": hpo_id, "pmcid": pmcid, "error": "missing chunk_key"}}
                    continue

                key = make_item_key("pmc", hpo_id, pmcid, chunk_key)
                if key in done_keys:
                    continue

                yield {
                    "kind": "job",
                    "doc_source": "pmc",
                    "doc_id": pmcid,
                    "pmcid": pmcid,
                    "hpo_id": hpo_id,
                    "cand": cand,
                    "chunk_key": chunk_key,
                    "trace": {"hpo_idx": h_i, "pmcid_idx": p_i, "cand_idx": c_i},
                }


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser("phaseB1_run_all_hpo (pubmed abs main)")

    ap.add_argument("--phaseA_run_dir", type=str, required=True)
    ap.add_argument("--prompt_txt", type=str, required=True)
    ap.add_argument("--out_base", type=str, required=True)
    ap.add_argument("--hpo_json", type=str, required=True)

    ap.add_argument("--doc_mode", type=str, default="hybrid", choices=["pubmed_only", "pmc_only", "hybrid"])

    # PubMed budgets (main)
    ap.add_argument("--pmids_per_hpo", type=int, default=5)
    ap.add_argument("--max_abs_chars", type=int, default=1600)

    # PMC budgets (optional in hybrid)
    ap.add_argument("--pmcids_per_hpo", type=int, default=2)
    ap.add_argument("--cands_per_pmcid", type=int, default=1)
    ap.add_argument("--max_candidates_per_pmcid", type=int, default=5)

    # HPO text budgets
    ap.add_argument("--max_def_chars", type=int, default=900)
    ap.add_argument("--max_llmdef_chars", type=int, default=700)

    # PMC candidate ctx budget
    ap.add_argument("--max_ctx_chars", type=int, default=1800)

    # B1 output control
    ap.add_argument("--b1_max_lines", type=int, default=8, help="Clamp max number of output lines stored (no min).")
    ap.add_argument("--none_token", type=str, default="NONE", help="Token that indicates no evidence (single-line output).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry_k", type=int, default=3)
    ap.add_argument("--retry_backoff", type=float, default=0.6)

    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")

    # resume / shard
    ap.add_argument("--resume", action="store_true", help="Resume from existing b1_outputs.jsonl in out_dir if present.")
    ap.add_argument("--out_dir", type=str, default="", help="If set, write into this dir and enable resume there.")
    ap.add_argument("--start", type=int, default=0, help="Shard start index (over filtered HPO list)")
    ap.add_argument("--end", type=int, default=-1, help="Shard end index (exclusive). -1 means end.")

    args = ap.parse_args()
    rng = random.Random(int(args.seed))

    phaseA_run_dir = args.phaseA_run_dir.rstrip("/")
    if not os.path.isdir(phaseA_run_dir):
        raise RuntimeError(f"--phaseA_run_dir not found: {phaseA_run_dir}")

    # Required PubMed files for pubmed_only/hybrid
    hpo_to_pmids_path = os.path.join(phaseA_run_dir, "pubmed", "hpo_to_pmids.jsonl")
    pmid_to_abs_path = os.path.join(phaseA_run_dir, "pubmed", "pmid_to_abstract.jsonl")

    if args.doc_mode in ("pubmed_only", "hybrid"):
        if not os.path.exists(hpo_to_pmids_path):
            raise RuntimeError(f"Missing hpo_to_pmids.jsonl: {hpo_to_pmids_path}")
        if not os.path.exists(pmid_to_abs_path):
            raise RuntimeError(f"Missing pmid_to_abstract.jsonl: {pmid_to_abs_path}")

    # Optional/required PMC files for pmc_only/hybrid
    merged_candidates_path = os.path.join(phaseA_run_dir, "phaseA", "merged", "mentions_candidates.jsonl")
    hpo_to_pmcids_path = os.path.join(phaseA_run_dir, "pubmed", "hpo_to_pmcids.jsonl")

    if args.doc_mode in ("pmc_only", "hybrid"):
        if not os.path.exists(hpo_to_pmcids_path):
            raise RuntimeError(f"Missing hpo_to_pmcids.jsonl: {hpo_to_pmcids_path}")
        if not os.path.exists(merged_candidates_path):
            raise RuntimeError(f"Missing merged candidates: {merged_candidates_path}")

    # Output dir
    if args.out_dir.strip():
        out_dir = args.out_dir.strip()
    else:
        run_id = os.path.basename(phaseA_run_dir.rstrip("/")) or "run"
        out_dir = os.path.join(args.out_base, f"B1_pubmed_main_{args.doc_mode}_{run_id}_{now_ts()}")
    ensure_dir(out_dir)

    outputs_path = os.path.join(out_dir, "b1_outputs.jsonl")
    failures_path = os.path.join(out_dir, "failures.jsonl")
    config_path = os.path.join(out_dir, "run_config.json")
    state_path = os.path.join(out_dir, "resume_state.json")

    # Write config ONCE
    if not os.path.exists(config_path):
        atomic_write_json(config_path, {
            "phaseA_run_dir": phaseA_run_dir,
            "prompt_txt": args.prompt_txt,
            "hpo_json": args.hpo_json,
            "doc_mode": args.doc_mode,

            "hpo_to_pmids_path": hpo_to_pmids_path if os.path.exists(hpo_to_pmids_path) else "",
            "pmid_to_abstract_path": pmid_to_abs_path if os.path.exists(pmid_to_abs_path) else "",

            "hpo_to_pmcids_path": hpo_to_pmcids_path if os.path.exists(hpo_to_pmcids_path) else "",
            "merged_candidates_path": merged_candidates_path if os.path.exists(merged_candidates_path) else "",

            "model": args.model,
            "base_url": args.base_url,
            "timeout": float(args.timeout),
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),

            "pmids_per_hpo": int(args.pmids_per_hpo),
            "max_abs_chars": int(args.max_abs_chars),

            "pmcids_per_hpo": int(args.pmcids_per_hpo),
            "cands_per_pmcid": int(args.cands_per_pmcid),
            "max_candidates_per_pmcid": int(args.max_candidates_per_pmcid),

            "max_ctx_chars": int(args.max_ctx_chars),
            "max_def_chars": int(args.max_def_chars),
            "max_llmdef_chars": int(args.max_llmdef_chars),

            "b1_max_lines": int(args.b1_max_lines),
            "none_token": args.none_token,

            "workers": int(args.workers),
            "retry_k": int(args.retry_k),
            "retry_backoff": float(args.retry_backoff),
            "seed": int(args.seed),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        })

    # Resume: load done set
    done_keys: Set[str] = set()
    if args.resume and file_exists_nonempty(outputs_path):
        print(f"[RESUME] loading done keys from: {outputs_path}")
        done_keys = load_done_keys(outputs_path)
        print(f"[RESUME] done_keys={len(done_keys)}")

    # Load prompt template
    prompt_tpl = load_prompt_template(args.prompt_txt)

    # Load HPO json (primary)
    hpo_meta, hpo_pool = load_hpo_json_primary(
        args.hpo_json,
        exclude_root=True,
        max_def_chars=int(args.max_def_chars),
        max_llmdef_chars=int(args.max_llmdef_chars),
    )

    # Load PubMed maps (if used)
    hpo_to_pmids: Dict[str, List[str]] = {}
    if args.doc_mode in ("pubmed_only", "hybrid"):
        hpo_to_pmids = load_hpo_to_pmids(hpo_to_pmids_path)

    # Load PMC maps (if used)
    hpo_to_pmcids: Dict[str, List[str]] = {}
    if args.doc_mode in ("pmc_only", "hybrid"):
        hpo_to_pmcids = load_hpo_to_pmcids(hpo_to_pmcids_path)

    # Filter HPO pool by mode coverage
    if args.doc_mode == "pubmed_only":
        filtered_hpos = [hid for hid in hpo_pool if hid in hpo_to_pmids and len(hpo_to_pmids[hid]) > 0]
    elif args.doc_mode == "pmc_only":
        filtered_hpos = [hid for hid in hpo_pool if hid in hpo_to_pmcids and len(hpo_to_pmcids[hid]) > 0]
    else:
        filtered_hpos = [
            hid for hid in hpo_pool
            if (hid in hpo_to_pmids and len(hpo_to_pmids[hid]) > 0)
            or (hid in hpo_to_pmcids and len(hpo_to_pmcids[hid]) > 0)
        ]

    if not filtered_hpos:
        raise RuntimeError("After filtering by doc_mode mappings, no HPO IDs remain.")

    # shard
    s = max(int(args.start), 0)
    e = int(args.end)
    if e < 0 or e > len(filtered_hpos):
        e = len(filtered_hpos)
    if s >= e:
        raise RuntimeError(f"Invalid shard range: start={s}, end={e}, total={len(filtered_hpos)}")
    shard_hpos = filtered_hpos[s:e]
    print(f"[INFO] HPO total={len(filtered_hpos)}, shard=[{s}:{e}) -> {len(shard_hpos)}")

    # ---- PubMed: build target_pmids (budgeted) and load abstracts filtered
    pmid_to_abs: Dict[str, Dict[str, Any]] = {}
    if args.doc_mode in ("pubmed_only", "hybrid"):
        target_pmids: Set[str] = set()
        for hid in shard_hpos:
            pmids = hpo_to_pmids.get(hid, [])
            if not pmids:
                continue
            k = min(int(args.pmids_per_hpo), len(pmids))
            chosen = rng.sample(pmids, k) if len(pmids) >= k else pmids
            for p in chosen:
                if p:
                    target_pmids.add(p)
        print(f"[INFO] target_pmids (shard, after budget) = {len(target_pmids)}")
        if target_pmids:
            pmid_to_abs = load_pmid_to_abstract_filtered(pmid_to_abs_path, target_pmids)
        print(f"[INFO] indexed pmid abstracts: {len(pmid_to_abs)}")

    # ---- PMC: build target_pmcids and candidates index filtered (optional)
    pmcid_to_cands: Dict[str, List[Dict[str, Any]]] = {}
    if args.doc_mode in ("pmc_only", "hybrid"):
        target_pmcids: Set[str] = set()
        for hid in shard_hpos:
            pmcids = hpo_to_pmcids.get(hid, [])
            if not pmcids:
                continue
            k_p = min(int(args.pmcids_per_hpo), len(pmcids))
            chosen = rng.sample(pmcids, k_p) if len(pmcids) >= k_p else pmcids
            for p in chosen:
                if p:
                    target_pmcids.add(p)
        print(f"[INFO] target_pmcids (shard, after budget) = {len(target_pmcids)}")
        if target_pmcids:
            pmcid_to_cands = build_pmcid_to_candidates_index_filtered(
                merged_candidates_path=merged_candidates_path,
                target_pmcids=target_pmcids,
                max_per_pmcid=int(args.max_candidates_per_pmcid),
                rng=rng,
            )
        print(f"[INFO] indexed pmcids with candidates: {len(pmcid_to_cands)}")

    # Init LLM
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Env {args.api_key_env} not set.")

    from Clients.llm_client import LLMClient  # type: ignore
    llm = LLMClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=float(args.timeout),
    )

    system_msg = "You are a biomedical text mining assistant."

    # Build job list
    job_list: List[Dict[str, Any]] = []
    build_failures: List[Dict[str, Any]] = []

    # PubMed jobs first (main)
    if args.doc_mode in ("pubmed_only", "hybrid"):
        for x in tqdm(
            iter_jobs_pubmed(
                hpo_ids=shard_hpos,
                hpo_meta=hpo_meta,
                hpo_to_pmids=hpo_to_pmids,
                pmid_to_abs=pmid_to_abs,
                pmids_per_hpo=int(args.pmids_per_hpo),
                max_abs_chars=int(args.max_abs_chars),
                rng=rng,
                done_keys=done_keys,
            ),
            desc="Build jobs (PubMed)",
            unit="job",
            total=None,
        ):
            if x.get("kind") == "fail":
                build_failures.append(x["row"])
            else:
                job_list.append(x)

    # Optional PMC jobs
    if args.doc_mode in ("pmc_only", "hybrid"):
        for x in tqdm(
            iter_jobs_pmc_optional(
                hpo_ids=shard_hpos,
                hpo_meta=hpo_meta,
                hpo_to_pmcids=hpo_to_pmcids,
                pmcid_to_cands=pmcid_to_cands,
                pmcids_per_hpo=int(args.pmcids_per_hpo),
                cands_per_pmcid=int(args.cands_per_pmcid),
                rng=rng,
                done_keys=done_keys,
            ),
            desc="Build jobs (PMC)",
            unit="job",
            total=None,
        ):
            if x.get("kind") == "fail":
                build_failures.append(x["row"])
            else:
                job_list.append(x)

    if build_failures:
        append_jsonl(failures_path, (dict(r, ts=time.time()) for r in build_failures))

    print(f"[INFO] jobs={len(job_list)} (after resume skip)")

    # Worker function
    def worker(job: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Returns (output_row, failure_row)
        """
        doc_source = (job.get("doc_source") or "").strip()
        hpo_id = (job.get("hpo_id") or "").strip()
        trace = job.get("trace") or {}

        meta = hpo_meta.get(hpo_id, {})
        hpo_name = (meta.get("name") or "").strip()
        hpo_def = (meta.get("def") or "").strip()
        hpo_llm_def = (meta.get("llm_def") or "").strip()

        # Build context per source
        if doc_source == "pubmed":
            pmid = (job.get("pmid") or "").strip()
            doc_id = pmid
            chunk_key = job.get("chunk_key") or f"pmid:{pmid}"
            ctx = job.get("context_text") or ""
            evidence_level = "weak"
            abs_row = job.get("abs_row") or {}
            cand = None
            pmcid = None
        else:
            pmcid = (job.get("pmcid") or "").strip()
            doc_id = pmcid
            chunk_key = safe_str(job.get("chunk_key") or "").strip()
            cand = job.get("cand") or {}
            ctx = build_context_from_candidate(cand, max_ctx_chars=int(args.max_ctx_chars))
            evidence_level = "strong"
            abs_row = {}
            pmid = None

        prompt = fill_prompt(
            prompt_tpl,
            hpo_id=hpo_id,
            hpo_name=hpo_name,
            hpo_def=hpo_def,
            hpo_llm_def=hpo_llm_def,
            context=ctx
        )

        ok, text_or_err = run_one_item_with_retry(
            llm=llm,
            system_msg=system_msg,
            prompt=prompt,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            retry_k=int(args.retry_k),
            backoff_sec=float(args.retry_backoff),
        )

        # --- LLM failed (network / exception / empty after retries)
        if not ok:
            fail_row = {
                "stage": "llm_call_failed",
                "doc_source": doc_source,
                "doc_id": doc_id,
                "hpo_id": hpo_id,
                "hpo_name": hpo_name,
                "chunk_key": chunk_key,
                "error": text_or_err,
                "ts": time.time(),
                "trace": trace,
            }
            if doc_source == "pubmed":
                fail_row["pmid"] = pmid
            else:
                fail_row["pmcid"] = pmcid
            return None, fail_row

        # --- OK: sanitize lines
        lines = parse_b1_lines(text_or_err)
        lines = clamp_phrases(lines, max_lines=int(args.b1_max_lines))

        # --- For PubMed abstracts, require B1 lines to be anchored in the context
        if doc_source == "pubmed":
            lines = filter_b1_lines_by_context(lines, ctx)

        # --- Treat NONE (or empty after sanitation/anchoring) as NORMAL no-evidence outcome
        if len(lines) == 0 or is_none_output(lines, none_token=args.none_token):
            fail_row = {
                "stage": "no_evidence_none",
                "doc_source": doc_source,
                "doc_id": doc_id,
                "hpo_id": hpo_id,
                "hpo_name": hpo_name,
                "chunk_key": chunk_key,
                "error": args.none_token,
                "ts": time.time(),
                "trace": trace,
            }
            if doc_source == "pubmed":
                fail_row["pmid"] = pmid
            else:
                fail_row["pmcid"] = pmcid
            return None, fail_row

        out_row: Dict[str, Any] = {
            "doc_source": doc_source,
            "doc_id": doc_id,
            "evidence_level": evidence_level,

            "hpo_id": hpo_id,
            "hpo_name": hpo_name,
            "hpo_def": hpo_def,
            "hpo_llm_def": hpo_llm_def,
            "hpo_source": "hpo_json",

            "chunk_key": chunk_key,

            # raw + parsed
            "output_text": text_or_err,
            "output_lines": lines,

            "ts": time.time(),
            "trace": trace,
        }

        if doc_source == "pubmed":
            out_row["pmid"] = pmid
            out_row["title"] = abs_row.get("title") or abs_row.get("Title")
            out_row["abstract"] = abs_row.get("abstract") or abs_row.get("Abstract") or abs_row.get("abs")
            out_row["journal"] = abs_row.get("journal") or abs_row.get("Journal")
            out_row["year"] = abs_row.get("year") or abs_row.get("Year")
            out_row["authors"] = abs_row.get("authors") or abs_row.get("Authors")
            out_row["context"] = ctx
        else:
            out_row["pmcid"] = pmcid
            out_row["section"] = cand.get("section")
            out_row["para_idx"] = cand.get("para_idx")
            out_row["candidate_type"] = cand.get("type")
            out_row["surface"] = cand.get("surface")
            out_row["unit"] = cand.get("unit")
            out_row["label_options"] = cand.get("label_options")
            out_row["context"] = cand.get("context")

        return out_row, None

    # Run threaded
    n_out = 0
    n_fail = 0

    if job_list:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(worker, j) for j in job_list]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM (B1)", unit="req"):
                out_row, fail_row = fut.result()

                if out_row is not None:
                    append_jsonl(outputs_path, [out_row])
                    n_out += 1

                if fail_row is not None:
                    append_jsonl(failures_path, [fail_row])
                    n_fail += 1

                if (n_out + n_fail) % 200 == 0:
                    atomic_write_json(state_path, {
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "outputs_appended": n_out,
                        "failures_appended": n_fail,
                        "done_keys_loaded": len(done_keys),
                        "jobs_total": len(job_list),
                        "shard": {"start": s, "end": e, "size": len(shard_hpos)},
                        "doc_mode": args.doc_mode,
                    })

    atomic_write_json(state_path, {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "outputs_appended": n_out,
        "failures_appended": n_fail,
        "done_keys_loaded": len(done_keys),
        "jobs_total": len(job_list),
        "shard": {"start": s, "end": e, "size": len(shard_hpos)},
        "doc_mode": args.doc_mode,
        "done": True,
    })

    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] outputs appended={n_out} -> {outputs_path}")
    print(f"[DONE] failures appended={n_fail} -> {failures_path}")
    print(f"[DONE] config={config_path}")
    print(f"[DONE] state={state_path}")


if __name__ == "__main__":
    main()
