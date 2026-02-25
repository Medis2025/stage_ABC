#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_phaseB_B1_hpojson_primary.py  (REVISED: HPO JSON is the PRIMARY HPO_ID source)

You requested flow
------------------
1) Sample HPO IDs from --hpo_json (primary source)
2) For each sampled HPO ID, look up PMCIDs via:
     <phaseA_run_dir>/pubmed/hpo_to_pmcids.jsonl
   (NOTE: hpo_name in this jsonl is assumed empty/ignored)
3) Use those PMCIDs to index into merged mentions candidates:
     <phaseA_run_dir>/phaseA/merged/mentions_candidates.jsonl
   and build B1 prompts using contexts from candidates.

This avoids "pmcid has no candidates" by:
- Building a pmcid -> [candidate rows] index from merged jsonl (can be large; we cap per pmcid)
- Skipping HPOs whose PMCIDs have no candidates.

Usage
-----
python3 B1.py \
  --phaseA_run_dir "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/out_phaseA/20260126_164618" \
  --prompt_txt "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/prompts/B1.txt" \
  --out_base "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/out" \
  --hpo_json "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json" \
  --n_hpos 30 \
  --pmcids_per_hpo 3 \
  --cands_per_pmcid 1 \
  --seed 42 \
  --temperature 0.2 \
  --max_tokens 256

Env
---
export DEEPSEEK_API_KEY="..."

Outputs
-------
<out_dir>/
  - b1_outputs.jsonl
  - failures.jsonl
  - run_config.json

Notes
-----
- B1 prompt expects {HPO_ID}/{HPO_NAME}/{CONTEXT} or {{...}} placeholders.
- If not found, we append INPUT block at end.
- This script calls remote LLM via: from Clients.llm_client import LLMClient
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import random
from typing import Any, Dict, List, Optional, Iterable, Set, Tuple


# =============================================================================
# IO helpers
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

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def atomic_write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def guess_run_id(phaseA_run_dir: str) -> str:
    b = os.path.basename(phaseA_run_dir.rstrip("/"))
    return b or "run"


# =============================================================================
# Prompt fill
# =============================================================================

def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fill_prompt(tpl: str, hpo_id: str, hpo_name: str, context: str) -> str:
    rep_pairs = {
        "{HPO_ID}": hpo_id,
        "{HPO_NAME}": hpo_name,
        "{CONTEXT}": context,
        "{{HPO_ID}}": hpo_id,
        "{{HPO_NAME}}": hpo_name,
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
            + f"[HPO_NAME] {hpo_name}\n\n"
            + "[CONTEXT]\n"
            + context
            + "\n"
        )
    return out


# =============================================================================
# HPO JSON: primary HPO source
# =============================================================================

def load_hpo_id_to_name_and_pool(hpo_json_path: str,
                                 exclude_root: bool = True) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns:
      - hpo_id_to_name: hpo_id -> Name[0] (fallback Synonym[0] else "")
      - hpo_ids_pool: list of candidate hpo_ids to sample from
    """
    if not hpo_json_path or not os.path.exists(hpo_json_path):
        raise FileNotFoundError(f"--hpo_json not found: {hpo_json_path}")

    with open(hpo_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2name: Dict[str, str] = {}
    pool: List[str] = []

    if not isinstance(data, dict):
        raise ValueError("hpo_json must be a dict keyed by HPO IDs.")

    for hid, obj in data.items():
        hid = safe_str(hid).strip()
        if not hid.startswith("HP:"):
            continue
        if exclude_root and hid == "HP:0000001":
            continue
        if not isinstance(obj, dict):
            continue

        name = ""
        nm = obj.get("Name")
        syn = obj.get("Synonym")

        if isinstance(nm, list) and nm:
            name = safe_str(nm[0]).strip()
        elif isinstance(nm, str) and nm.strip():
            name = nm.strip()
        elif isinstance(syn, list) and syn:
            name = safe_str(syn[0]).strip()
        elif isinstance(syn, str) and syn.strip():
            name = syn.strip()

        id2name[hid] = name
        pool.append(hid)

    if not pool:
        raise RuntimeError("No usable HPO IDs found in hpo_json.")

    return id2name, pool


# =============================================================================
# hpo_to_pmcids.jsonl: map hpo_id -> pmcids
# =============================================================================

def load_hpo_to_pmcids(hpo_to_pmcids_path: str) -> Dict[str, List[str]]:
    """
    Build:
      hpo_id -> [pmcid, ...]
    Ignores hpo_name fields (assumed empty).
    """
    mp: Dict[str, List[str]] = {}
    for r in read_jsonl_iter(hpo_to_pmcids_path):
        hpo_id = (r.get("hpo_id") or "").strip()
        pmcids = r.get("pmcids") or []
        if not hpo_id or not isinstance(pmcids, list):
            continue
        out: List[str] = []
        seen: Set[str] = set()
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


# =============================================================================
# merged mentions candidates: pmcid -> candidates index (capped)
# =============================================================================

def build_pmcid_to_candidates_index(merged_candidates_path: str,
                                   max_per_pmcid: int,
                                   rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a memory-friendly index:
      pmcid -> up to max_per_pmcid candidate rows

    Strategy:
    - For each pmcid, we keep at most max_per_pmcid rows by simple reservoir.
    """
    max_per_pmcid = int(max_per_pmcid)
    if max_per_pmcid <= 0:
        raise ValueError("--max_candidates_per_pmcid must be > 0")

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}

    for row in read_jsonl_iter(merged_candidates_path):
        pmcid = safe_str(row.get("pmcid") or "").strip()
        if not pmcid:
            continue

        counts[pmcid] = counts.get(pmcid, 0) + 1
        n = counts[pmcid]

        if pmcid not in buckets:
            buckets[pmcid] = [row]
            continue

        bucket = buckets[pmcid]
        if len(bucket) < max_per_pmcid:
            bucket.append(row)
        else:
            # reservoir replace with probability max_per_pmcid / n
            j = rng.randint(1, n)  # 1..n
            if j <= max_per_pmcid:
                bucket[j - 1] = row

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
# LLM call wrapper (plain text)
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


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser("test_phaseB_B1_hpojson_primary")

    ap.add_argument("--phaseA_run_dir", type=str, required=True)
    ap.add_argument("--prompt_txt", type=str, required=True)
    ap.add_argument("--out_base", type=str, required=True)
    ap.add_argument("--hpo_json", type=str, required=True)

    ap.add_argument("--n_hpos", type=int, default=30, help="How many HPO IDs to sample from hpo_json")
    ap.add_argument("--pmcids_per_hpo", type=int, default=3, help="How many PMCIDs to try per HPO")
    ap.add_argument("--cands_per_pmcid", type=int, default=1, help="How many candidates to run per PMCID")

    ap.add_argument("--max_candidates_per_pmcid", type=int, default=5,
                    help="Cap candidate rows stored per pmcid when indexing merged jsonl (tradeoff memory vs recall)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_ctx_chars", type=int, default=1800)

    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")

    args = ap.parse_args()
    rng = random.Random(int(args.seed))

    phaseA_run_dir = args.phaseA_run_dir.rstrip("/")
    if not os.path.isdir(phaseA_run_dir):
        raise RuntimeError(f"--phaseA_run_dir not found: {phaseA_run_dir}")

    merged_candidates_path = os.path.join(phaseA_run_dir, "phaseA", "merged", "mentions_candidates.jsonl")
    if not os.path.exists(merged_candidates_path):
        raise RuntimeError(f"Missing merged candidates: {merged_candidates_path}")

    hpo_to_pmcids_path = os.path.join(phaseA_run_dir, "pubmed", "hpo_to_pmcids.jsonl")
    if not os.path.exists(hpo_to_pmcids_path):
        raise RuntimeError(f"Missing hpo_to_pmcids.jsonl: {hpo_to_pmcids_path}")

    # 1) HPO JSON is primary
    hpo_id_to_name, hpo_pool = load_hpo_id_to_name_and_pool(args.hpo_json, exclude_root=True)

    # 2) hpo_id -> pmcids
    hpo_to_pmcids = load_hpo_to_pmcids(hpo_to_pmcids_path)

    # Filter pool to those having pmcids
    hpo_pool = [hid for hid in hpo_pool if hid in hpo_to_pmcids and len(hpo_to_pmcids[hid]) > 0]
    if not hpo_pool:
        raise RuntimeError("After filtering by hpo_to_pmcids.jsonl, no HPO IDs remain.")

    # sample HPO IDs
    n_hpos = min(int(args.n_hpos), len(hpo_pool))
    sampled_hpos = rng.sample(hpo_pool, n_hpos)

    # 3) Index merged mentions by pmcid (capped)
    pmcid_to_cands = build_pmcid_to_candidates_index(
        merged_candidates_path,
        max_per_pmcid=int(args.max_candidates_per_pmcid),
        rng=rng
    )

    # Load prompt template
    prompt_tpl = load_prompt_template(args.prompt_txt)

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

    # Output dir
    run_id = guess_run_id(phaseA_run_dir)
    out_dir = os.path.join(args.out_base, f"test_B1_hpojson_primary_{run_id}_{now_ts()}")
    ensure_dir(out_dir)

    system_msg = "You are a biomedical text mining assistant."

    outputs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # Iterate HPOs -> PMCIDs -> candidates
    for h_i, hpo_id in enumerate(sampled_hpos, start=1):
        hpo_name = (hpo_id_to_name.get(hpo_id) or "").strip()
        pmcids = list(hpo_to_pmcids.get(hpo_id, []))

        if not pmcids:
            failures.append({"stage": "no_pmcids_for_hpo", "hpo_id": hpo_id, "hpo_name": hpo_name})
            continue

        # pick subset of pmcids
        k_p = min(int(args.pmcids_per_hpo), len(pmcids))
        chosen_pmcids = rng.sample(pmcids, k_p) if len(pmcids) >= k_p else pmcids

        for p_i, pmcid in enumerate(chosen_pmcids, start=1):
            cand_list = pmcid_to_cands.get(pmcid, [])
            if not cand_list:
                failures.append({
                    "stage": "no_candidates_for_pmcid",
                    "hpo_id": hpo_id,
                    "hpo_name": hpo_name,
                    "pmcid": pmcid
                })
                continue

            k_c = min(int(args.cands_per_pmcid), len(cand_list))
            chosen_cands = rng.sample(cand_list, k_c) if len(cand_list) >= k_c else cand_list

            for c_i, cand in enumerate(chosen_cands, start=1):
                ctx = build_context_from_candidate(cand, max_ctx_chars=int(args.max_ctx_chars))
                prompt = fill_prompt(prompt_tpl, hpo_id=hpo_id, hpo_name=hpo_name, context=ctx)

                try:
                    text = call_llm_plain_text(
                        llm,
                        system=system_msg,
                        user=prompt,
                        temperature=float(args.temperature),
                        max_tokens=int(args.max_tokens) if args.max_tokens else None,
                    )

                    outputs.append({
                        "hpo_id": hpo_id,
                        "hpo_name": hpo_name,
                        "hpo_name_source": "hpo_json",
                        "pmcid": pmcid,

                        "chunk_key": cand.get("chunk_key"),
                        "section": cand.get("section"),
                        "para_idx": cand.get("para_idx"),
                        "candidate_type": cand.get("type"),
                        "surface": cand.get("surface"),
                        "unit": cand.get("unit"),
                        "label_options": cand.get("label_options"),
                        "context": cand.get("context"),

                        "prompt_path": args.prompt_txt,
                        "hpo_json": args.hpo_json,
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": float(args.temperature),
                        "max_tokens": int(args.max_tokens),
                        "max_ctx_chars": int(args.max_ctx_chars),

                        "output_text": (text or "").strip(),
                        "ts": time.time(),

                        "trace": {
                            "hpo_idx": h_i,
                            "pmcid_idx": p_i,
                            "cand_idx": c_i,
                            "n_hpos": n_hpos,
                            "pmcids_per_hpo": int(args.pmcids_per_hpo),
                            "cands_per_pmcid": int(args.cands_per_pmcid),
                            "max_candidates_per_pmcid": int(args.max_candidates_per_pmcid),
                        }
                    })

                except Exception as e:
                    failures.append({
                        "stage": "llm_call",
                        "hpo_id": hpo_id,
                        "hpo_name": hpo_name,
                        "pmcid": pmcid,
                        "chunk_key": cand.get("chunk_key"),
                        "error": str(e),
                        "ts": time.time(),
                    })

    write_jsonl(os.path.join(out_dir, "b1_outputs.jsonl"), outputs)
    write_jsonl(os.path.join(out_dir, "failures.jsonl"), failures)
    atomic_write_json(os.path.join(out_dir, "run_config.json"), {
        "phaseA_run_dir": phaseA_run_dir,
        "merged_candidates_path": merged_candidates_path,
        "hpo_to_pmcids_path": hpo_to_pmcids_path,
        "prompt_txt": args.prompt_txt,
        "hpo_json": args.hpo_json,
        "out_dir": out_dir,
        "args": vars(args),
        "n_hpo_pool_after_filter": len(hpo_pool),
        "n_hpos_sampled": len(sampled_hpos),
        "n_outputs": len(outputs),
        "n_failures": len(failures),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    })

    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] outputs={len(outputs)} -> {os.path.join(out_dir, 'b1_outputs.jsonl')}")
    print(f"[DONE] failures={len(failures)} -> {os.path.join(out_dir, 'failures.jsonl')}")


if __name__ == "__main__":
    main()
