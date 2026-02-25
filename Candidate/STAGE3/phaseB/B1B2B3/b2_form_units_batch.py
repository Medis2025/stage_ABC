#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_medium_embedding_pool.py  (FIXED for your b2_units.jsonl)

What this does
--------------
Build a "medium evidence embedding pool" from B2 outputs (b2_units.jsonl):

Input (your current schema):
- Each JSONL line is a *support unit* with:
    - hpo_id
    - support_sentence / canonical_phrase
    - from_b1.output_lines (list[str])  <-- present but NOT required

We treat "medium lines" as:
  1) canonical_phrase
  2) support_sentence
  3) (optional) from_b1.output_lines

Then we embed ALL kept lines into a flattened pool:
  - E_med_lines.npy: [L, D] float32, L2-normalized
And build alignment/index for each HPO in master order:
  - med_index.json:  hpo_id -> [start, end)
  - mask_med.npy:    [N] uint8, 1 if this HPO has any medium lines
  - med_source.jsonl: per-HPO stats (raw/kept/start/end)
  - stats_med.json:  global stats, key hits, coverage, etc.

Why the previous version produced all zeros
-------------------------------------------
Your b2_units.jsonl stores candidate lines under:
  from_b1.output_lines
and also has canonical_phrase/support_sentence at top-level.

If extract_lines() didn't read these, you'd get:
  NOT_FOUND high, coverage=0, quantiles all 0

This version:
- Prioritizes canonical_phrase + support_sentence
- Also collects from_b1.output_lines when present
- Dedup + clamp + min_len filtering
- Atomic writes for .npy and json/jsonl (prevents os.replace tmp missing)

Outputs
-------
Saved under:
  <base_out_dir>/<subdir>/
Default: .../B2_embed/out/Medium_embed/

Run command (YOUR REQUESTED)
---------------------------
python3 /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/build_medium_embedding_pool.py \
  --b2_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/out/B2/B2_20260202_173225/b2_units.jsonl \
  --base_out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out \
  --subdir Medium_embed \
  --master_hpo_ids /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out/Def_embed/hpo_ids.json \
  --dedup

Tip:
- Add --dry_run first to validate schema/coverage quickly.
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


# -------------------------
# Defaults (safe)
# -------------------------

DEFAULT_MODEL_DIR = "/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B"

DEFAULT_B2_PATH = (
    "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/"
    "pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/"
    "phaseB/B1B2B3/out/B2/B2_20260202_173225/b2_units.jsonl"
)

DEFAULT_BASE_OUT_DIR = (
    "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/"
    "pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/"
    "phaseB/B1B2B3/B2_embed/out"
)

_WS = re.compile(r"\s+")


# -------------------------
# IO helpers (atomic)
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def write_json_atomic(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def write_jsonl_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """
    Atomic save for .npy:
    IMPORTANT: tmp must END WITH .npy, otherwise np.save will append ".npy"
    and os.replace will fail.
    """
    if not path.endswith(".npy"):
        raise ValueError(f"Output npy path must end with .npy: {path}")
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


# -------------------------
# text utils
# -------------------------

def clean(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s

def nonempty_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, list):
        x = [str(x)]
    out: List[str] = []
    for t in x:
        tt = clean(str(t))
        if tt:
            out.append(tt)
    return out

def l2_normalize_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if (not np.isfinite(n)) or n <= 0:
        return np.zeros_like(v, dtype=np.float32)
    return (v / (n + 1e-12)).astype(np.float32)

def quantiles_int(xs: List[int], qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": 0.0 for q in qs} | {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(xs, dtype=np.float32)
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(arr, q))
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["mean"] = float(arr.mean())
    return out


# -------------------------
# jsonl loading
# -------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def get_hpo_id(obj: Dict[str, Any]) -> str:
    for k in ["hpo_id", "Id", "id", "HPO_ID", "hp_id"]:
        v = obj.get(k)
        if isinstance(v, str) and v.startswith("HP:"):
            return v.strip()
    return ""


# -------------------------
# Schema-aware extraction (KEY FIX)
# -------------------------

def extract_medium_lines(obj: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Extract "medium lines" from one B2 unit.

    We use:
      - canonical_phrase (top-level)
      - support_sentence (top-level)
      - from_b1.output_lines (optional phrase candidates)

    Return (lines, source_tag_for_stats)
    """
    lines: List[str] = []

    canonical = clean(str(obj.get("canonical_phrase") or ""))
    support = clean(str(obj.get("support_sentence") or ""))

    if canonical:
        lines.append(canonical)
    if support:
        lines.append(support)

    fb1 = obj.get("from_b1")
    if isinstance(fb1, dict):
        # in your schema, this exists
        fb1_lines = nonempty_list(fb1.get("output_lines"))
        if fb1_lines:
            lines.extend(fb1_lines)

    if lines:
        return lines, "canonical+support+from_b1.output_lines"

    # fallbacks (just in case)
    for k in ["output_lines", "lines", "evidence_lines", "medium_lines"]:
        if k in obj:
            xs = nonempty_list(obj.get(k))
            if xs:
                return xs, k

    return [], "NOT_FOUND"


def build_hpo_to_lines(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Counter]:
    """
    Map hpo_id -> concatenated medium lines across all occurrences.
    Also count which key-path was used.
    """
    hpo2lines: Dict[str, List[str]] = {}
    key_hits = Counter()

    for obj in rows:
        if not isinstance(obj, dict):
            continue
        hid = get_hpo_id(obj)
        if not hid:
            continue
        lines, src = extract_medium_lines(obj)
        key_hits[src] += 1
        if not lines:
            continue
        hpo2lines.setdefault(hid, []).extend(lines)

    return hpo2lines, key_hits


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser("build_medium_embedding_pool (B2 units -> embedding pool)")
    ap.add_argument("--b2_path", type=str, default=DEFAULT_B2_PATH, help="b2_units.jsonl path")
    ap.add_argument("--base_out_dir", type=str, default=DEFAULT_BASE_OUT_DIR, help="Base output dir")
    ap.add_argument("--subdir", type=str, default="Medium_embed", help="Subdir under base_out_dir")

    ap.add_argument("--master_hpo_ids", type=str, required=True, help="Path to Def_embed/hpo_ids.json (master order)")

    ap.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / cuda:0 ... (empty=auto)")
    ap.add_argument("--dtype", type=str, default="", help="bf16/fp16/fp32 (empty=auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--max_lines_per_hpo", type=int, default=50, help="Clamp per HPO after cleaning/dedup (0=no clamp)")
    ap.add_argument("--min_line_len", type=int, default=3, help="Drop very short lines")
    ap.add_argument("--dedup", action="store_true", help="Dedup lines per HPO")
    ap.add_argument("--dry_run", action="store_true", help="Parse + stats only, do not embed")

    args = ap.parse_args()

    out_dir = os.path.join(args.base_out_dir, args.subdir)
    ensure_dir(out_dir)

    if not os.path.exists(args.master_hpo_ids):
        raise FileNotFoundError(f"master_hpo_ids not found: {args.master_hpo_ids}")

    with open(args.master_hpo_ids, "r", encoding="utf-8") as f:
        master = json.load(f)

    hpo_ids = master.get("ids", [])
    if not isinstance(hpo_ids, list) or not hpo_ids or not all(isinstance(x, str) for x in hpo_ids):
        raise ValueError("master_hpo_ids.json must be like: {\"ids\": [\"HP:...\", ...]}")

    N = len(hpo_ids)

    # Load B2 units
    rows = load_jsonl(args.b2_path)
    hpo2lines_raw, key_hits = build_hpo_to_lines(rows)

    # Prepare outputs
    out_E = os.path.join(out_dir, "E_med_lines.npy")
    out_index = os.path.join(out_dir, "med_index.json")
    out_mask = os.path.join(out_dir, "mask_med.npy")
    out_meta = os.path.join(out_dir, "med_source.jsonl")
    out_stats = os.path.join(out_dir, "stats_med.json")

    mask_med = np.zeros((N,), dtype=np.uint8)
    med_index: Dict[str, List[int]] = {}
    meta_rows: List[Dict[str, Any]] = []

    raw_counts: List[int] = []
    kept_counts: List[int] = []

    # If dry_run, we only advance cursor by kept, no embedding is computed.
    cursor = 0
    all_vecs: List[np.ndarray] = []

    client = None
    if not args.dry_run:
        client = Qwen3EmbeddingClient(EmbeddingConfig(
            model_dir=args.model_dir,
            device=(args.device or None),
            dtype=(args.dtype or None),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            normalize=True,
            use_instruct=False,          # doc-like text
            attn_implementation=None,
        ))

    p = tqdm(range(N), desc="MED align+embed", unit="hpo")
    for row_idx in p:
        hid = hpo_ids[row_idx]
        raw_lines = hpo2lines_raw.get(hid, []) or []
        raw_counts.append(len(raw_lines))

        # clean + min len
        lines = [clean(x) for x in raw_lines]
        lines = [x for x in lines if x and len(x) >= int(args.min_line_len)]

        # dedup
        if args.dedup and lines:
            seen = set()
            uniq = []
            for x in lines:
                if x in seen:
                    continue
                seen.add(x)
                uniq.append(x)
            lines = uniq

        # clamp
        if int(args.max_lines_per_hpo) > 0 and len(lines) > int(args.max_lines_per_hpo):
            lines = lines[: int(args.max_lines_per_hpo)]

        kept = len(lines)
        kept_counts.append(kept)

        start = cursor
        end = cursor

        if kept > 0:
            mask_med[row_idx] = 1
            if args.dry_run:
                end = start + kept
                cursor = end
            else:
                assert client is not None
                emb = client.encode(lines, mode="doc", return_numpy=True)  # [m,D], already L2 norm-ish
                emb = np.asarray(emb, dtype=np.float32)
                for i in range(emb.shape[0]):
                    v = l2_normalize_f32(emb[i])
                    if not np.isfinite(v).all():
                        raise RuntimeError(f"Non-finite embedding at hpo={hid} row={row_idx} line={i}")
                    all_vecs.append(v)
                end = start + emb.shape[0]
                cursor = end

        med_index[hid] = [int(start), int(end)]
        meta_rows.append({
            "hpo_id": hid,
            "row_idx": int(row_idx),
            "raw_lines": int(len(raw_lines)),
            "kept_lines": int(kept),
            "start": int(start),
            "end": int(end),
        })
        p.set_postfix({"L_total": cursor, "hit": int(mask_med[row_idx])})
    p.close()

    # Save alignment artifacts
    write_json_atomic(out_index, med_index)
    atomic_save_npy(out_mask, mask_med.astype(np.uint8))
    write_jsonl_atomic(out_meta, meta_rows)

    # Dry run stats (no embeddings)
    if args.dry_run:
        stats = {
            "created_at": now_str(),
            "mode": "dry_run",
            "b2_path": args.b2_path,
            "master_hpo_ids": args.master_hpo_ids,
            "base_out_dir": args.base_out_dir,
            "out_dir": out_dir,
            "N": N,
            "key_hits": dict(key_hits),
            "raw_lines_quantiles": quantiles_int(raw_counts),
            "kept_lines_quantiles": quantiles_int(kept_counts),
            "coverage_hpo_with_medium": int(mask_med.sum()),
            "files": {
                "med_index": out_index,
                "mask_med": out_mask,
                "meta": out_meta,
                "stats": out_stats,
            },
            "notes": [
                "If key_hits is dominated by canonical+support+from_b1.output_lines, schema is correct.",
                "If NOT_FOUND is high or coverage==0, inspect b2_units.jsonl schema.",
            ],
        }
        write_json_atomic(out_stats, stats)
        tqdm.write("[DONE] DRY RUN finished.")
        tqdm.write(f"  stats: {out_stats}")
        return

    # Save embedding pool
    if all_vecs:
        E = np.stack(all_vecs, axis=0).astype(np.float32)
    else:
        # should not happen if coverage > 0, but keep safe
        E = np.zeros((0, 4096), dtype=np.float32)

    atomic_save_npy(out_E, E)

    stats = {
        "created_at": now_str(),
        "b2_path": args.b2_path,
        "master_hpo_ids": args.master_hpo_ids,
        "base_out_dir": args.base_out_dir,
        "out_dir": out_dir,
        "N": N,
        "L_total_lines_embedded": int(E.shape[0]),
        "D": int(E.shape[1]) if E.ndim == 2 else 0,
        "coverage_hpo_with_medium": int(mask_med.sum()),
        "raw_lines_quantiles": quantiles_int(raw_counts),
        "kept_lines_quantiles": quantiles_int(kept_counts),
        "key_hits": dict(key_hits),
        "files": {
            "E_med_lines": out_E,
            "med_index": out_index,
            "mask_med": out_mask,
            "meta": out_meta,
            "stats": out_stats,
        },
        "notes": [
            "E_med_lines is flattened pool of medium lines.",
            "med_index maps hpo_id -> [start,end).",
            "mask_med aligned to master HPO id list.",
            "Line sources used: canonical_phrase + support_sentence + from_b1.output_lines (when present).",
        ],
    }
    write_json_atomic(out_stats, stats)

    tqdm.write("[DONE] Medium embedding pool saved.")
    tqdm.write(f"  E_med_lines: {out_E}")
    tqdm.write(f"  stats: {out_stats}")


if __name__ == "__main__":
    main()
