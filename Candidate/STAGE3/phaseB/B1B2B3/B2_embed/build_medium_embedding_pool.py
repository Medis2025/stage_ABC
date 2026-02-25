#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_medium_embedding_pool.py  (SAVE TO out/Medium_embed)

Input (default):
  /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/out/B2/B2_20260202_173225/b2_units.jsonl

Base out dir (default, created if missing):
  /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out

THIS SCRIPT writes into:
  <base_out_dir>/Medium_embed

Outputs:
- E_med_lines.npy   [L, D] float32, unit norm (CPU exact normalize)
- med_index.json    {hpo_id: [start,end), ...}
- mask_med.npy      [N] uint8 aligned to master hpo_ids.json order
- med_source.jsonl  per HPO stats
- stats_med.json    global stats

Alignment:
- Uses master_hpo_ids.json (default: <base_out_dir>/hpo_ids.json from DEF step).
  You can pass a different master file via --master_hpo_ids.

Key FIX:
- atomic_save_npy() ensures tmp path endswith ".npy" (avoids np.save auto-appending).

REVISION (2026-02-03)
---------------------
✅ Fix schema mismatch for your "units jsonl" example:
   - lines are located at: from_b1.output_lines
✅ Add counters in stats:
   - rows_total / rows_with_hid / rows_with_hid_and_lines / rows_with_any_lines
✅ Remove hard-coded empty embedding dim (4096); infer D from first embed
✅ Add schema warning when NOT_FOUND ratio is high
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from tqdm import tqdm

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


DEFAULT_BASE_OUT_DIR = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out"
DEFAULT_MODEL_DIR = "/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B"

DEFAULT_B2_PATH = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/out/B2/B2_20260202_173225/b2_units.jsonl"


# -------------------------
# small utils
# -------------------------

_WS = re.compile(r"\s+")
_HP_RE = re.compile(r"(HP[:_-]?\s*\d{7})", re.I)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_save_npy(final_path: str, arr: np.ndarray) -> None:
    """
    Atomic save for .npy:
    IMPORTANT: tmp must end with .npy, otherwise np.save auto-appends ".npy"
    """
    if not final_path.endswith(".npy"):
        raise ValueError(f"final_path must end with .npy: {final_path}")
    tmp_path = final_path + ".tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, final_path)


def clean_line(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s


def normalize_hp(s: str) -> str:
    """
    Normalize common HPO ID formats:
    - HP:0001250
    - HP_0001250
    - HP-0001250
    - 0001250
    """
    s = (s or "").strip()
    if not s:
        return ""
    if s.isdigit() and len(s) == 7:
        return f"HP:{s}"
    m = _HP_RE.search(s)
    if not m:
        return ""
    x = m.group(1).upper().strip()
    x = x.replace("HP_", "HP:").replace("HP-", "HP:").replace("HP :", "HP:")
    # ensure digits
    num = re.sub(r"\D", "", x)
    if len(num) == 7:
        return f"HP:{num}"
    return x if x.startswith("HP:") else ""


def nonempty_list(x: Any) -> List[str]:
    """
    Convert x to list[str], supporting:
    - str
    - list[str]
    - list[dict] with common text keys
    """
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, list):
        x = [x]
    out: List[str] = []
    for t in x:
        if isinstance(t, dict):
            # try common fields in dict items
            for kk in ["text", "line_text", "line", "surface", "value_text", "span_text"]:
                if kk in t and t[kk]:
                    t = t[kk]
                    break
        tt = clean_line(str(t))
        if tt:
            out.append(tt)
    return out


def l2_normalize_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0:
        return np.zeros_like(v, dtype=np.float32)
    return (v / (n + 1e-12)).astype(np.float32)


def quantiles_int(xs: List[int], qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": 0.0 for q in qs} | {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(xs, dtype=np.float32)
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(arr, q))
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["mean"] = float(arr.mean())
    return out


# -------------------------
# input loading
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
    # top-level
    for k in ["hpo_id", "Id", "id", "HPO_ID", "hp_id", "hpo", "hp", "hid", "HPO"]:
        v = obj.get(k)
        if isinstance(v, str):
            hid = normalize_hp(v)
            if hid:
                return hid
    # nested HPO dict
    if "HPO" in obj and isinstance(obj["HPO"], dict):
        for k in ["hpo_id", "Id", "id", "HPO_ID", "hp_id", "hpo", "hp", "hid"]:
            v = obj["HPO"].get(k)
            if isinstance(v, str):
                hid = normalize_hp(v)
                if hid:
                    return hid
    return ""


def extract_lines(obj: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Return (lines, source_key_used) by auto-detecting common keys.

    Revised to support your actual schema:
      - from_b1.output_lines
    """
    # 1) top-level candidates
    candidates = [
        "output_lines",
        "output_lines_sanitized",
        "lines",
        "evidence_lines",
        "medium_lines",
        "b2_lines",
        "unit_lines",
        "b2_units",
        "units",
    ]
    for k in candidates:
        if k in obj:
            lines = nonempty_list(obj.get(k))
            return lines, k

    # 2) common nested dicts (B2/b2)
    for top in ["B2", "b2"]:
        if top in obj and isinstance(obj[top], dict):
            b2 = obj[top]
            for k in ["output_lines", "output_lines_sanitized", "lines", "unit_lines", "b2_units", "units", "medium_lines"]:
                if k in b2:
                    lines = nonempty_list(b2.get(k))
                    return lines, f"{top}.{k}"

    # 3) NEW: your schema (from_b1.output_lines)
    if "from_b1" in obj and isinstance(obj["from_b1"], dict):
        fb1 = obj["from_b1"]
        for k in ["output_lines", "output_lines_sanitized", "lines"]:
            if k in fb1:
                lines = nonempty_list(fb1.get(k))
                return lines, f"from_b1.{k}"

    # optional alt naming
    if "from_B1" in obj and isinstance(obj["from_B1"], dict):
        fb1 = obj["from_B1"]
        for k in ["output_lines", "output_lines_sanitized", "lines"]:
            if k in fb1:
                lines = nonempty_list(fb1.get(k))
                return lines, f"from_B1.{k}"

    return [], "NOT_FOUND"


def build_hpo_to_lines(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Counter, Dict[str, int]]:
    """
    Map hpo_id -> concatenated list of lines across all occurrences.
    Also count which key was used for lines.

    Returns:
      - hpo2lines
      - key_hits (only for rows where hid exists)
      - counters dict:
          rows_total
          rows_with_hid
          rows_with_hid_and_lines (lines key found, may be empty list)
          rows_with_any_lines (lines key found and non-empty)
          rows_with_hid_but_no_lines_key (NOT_FOUND)
    """
    hpo2lines: Dict[str, List[str]] = {}
    key_hits = Counter()

    rows_total = 0
    rows_with_hid = 0
    rows_with_hid_and_lines = 0
    rows_with_any_lines = 0
    rows_with_hid_but_no_lines_key = 0

    for obj in rows:
        rows_total += 1
        if not isinstance(obj, dict):
            continue

        hid = get_hpo_id(obj)
        if not hid:
            continue
        rows_with_hid += 1

        lines, key_used = extract_lines(obj)
        key_hits[key_used] += 1

        if key_used != "NOT_FOUND":
            rows_with_hid_and_lines += 1
            if lines:
                rows_with_any_lines += 1
        else:
            rows_with_hid_but_no_lines_key += 1

        if not lines:
            continue

        hpo2lines.setdefault(hid, []).extend(lines)

    counters = {
        "rows_total": rows_total,
        "rows_with_hid": rows_with_hid,
        "rows_with_hid_and_lines": rows_with_hid_and_lines,
        "rows_with_any_lines": rows_with_any_lines,
        "rows_with_hid_but_no_lines_key": rows_with_hid_but_no_lines_key,
    }
    return hpo2lines, key_hits, counters


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser("build_medium_embedding_pool (save to out/Medium_embed)")
    ap.add_argument("--b2_path", type=str, default=DEFAULT_B2_PATH, help="B2 medium jsonl path")

    ap.add_argument(
        "--base_out_dir",
        type=str,
        default=DEFAULT_BASE_OUT_DIR,
        help="Base output dir (script will create <base_out_dir>/Medium_embed)",
    )
    ap.add_argument(
        "--subdir",
        type=str,
        default="Medium_embed",
        help="Subdirectory under base_out_dir (default=Medium_embed)",
    )
    ap.add_argument(
        "--master_hpo_ids",
        type=str,
        default="",
        help="Master hpo_ids.json (row order) to align mask_med.npy. Default=<base_out_dir>/hpo_ids.json",
    )

    ap.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / cuda:0 ... (empty=auto)")
    ap.add_argument("--dtype", type=str, default="", help="bf16/fp16/fp32 (empty=auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--max_lines_per_hpo", type=int, default=50, help="Clamp lines per HPO after cleaning/dedup (0 = no clamp)")
    ap.add_argument("--min_line_len", type=int, default=3, help="Drop very short lines")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate lines per HPO (recommended)")
    ap.add_argument("--dry_run", action="store_true", help="Parse + stats only, do not embed")

    args = ap.parse_args()

    out_dir = os.path.join(args.base_out_dir, args.subdir)
    ensure_dir(out_dir)

    master_path = args.master_hpo_ids.strip() or os.path.join(args.base_out_dir, "hpo_ids.json")
    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"master_hpo_ids not found: {master_path}\n"
            f"Tip: run DEF step first (it writes hpo_ids.json into base_out_dir), or pass --master_hpo_ids /path/to/hpo_ids.json"
        )

    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    hpo_ids = master.get("ids", [])
    if not isinstance(hpo_ids, list) or not hpo_ids or not all(isinstance(x, str) for x in hpo_ids):
        raise ValueError("master_hpo_ids.json must be like: {\"ids\": [\"HP:...\", ...]}")

    N = len(hpo_ids)

    # Load jsonl and map
    rows = load_jsonl(args.b2_path)
    hpo2lines_raw, key_hits, row_counters = build_hpo_to_lines(rows)

    # Init embedding client unless dry_run
    client = None
    if not args.dry_run:
        client = Qwen3EmbeddingClient(EmbeddingConfig(
            model_dir=args.model_dir,
            device=(args.device or None),
            dtype=(args.dtype or None),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            normalize=True,
            use_instruct=False,
            attn_implementation=None,
        ))

    # outputs (under out/Medium_embed)
    out_lines_npy = os.path.join(out_dir, "E_med_lines.npy")
    out_index = os.path.join(out_dir, "med_index.json")
    out_mask = os.path.join(out_dir, "mask_med.npy")
    out_meta = os.path.join(out_dir, "med_source.jsonl")
    out_stats = os.path.join(out_dir, "stats_med.json")

    mask_med = np.zeros((N,), dtype=np.uint8)
    med_index: Dict[str, List[int]] = {}
    meta_rows: List[Dict[str, Any]] = []

    all_vecs: List[np.ndarray] = []
    cursor = 0

    kept_counts: List[int] = []
    raw_counts: List[int] = []

    inferred_D: Optional[int] = None

    p = tqdm(range(N), desc="MED align+embed", unit="hpo")
    for row_idx in p:
        hid = hpo_ids[row_idx]
        raw_lines = hpo2lines_raw.get(hid, []) or []
        raw_counts.append(len(raw_lines))

        # clean + filter
        lines = [clean_line(x) for x in raw_lines]
        lines = [x for x in lines if x and len(x) >= int(args.min_line_len)]

        # dedup keep order
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

            if not args.dry_run:
                assert client is not None
                emb = client.encode(lines, mode="doc", return_numpy=True)  # [m, D]
                emb = np.asarray(emb, dtype=np.float32)

                if emb.ndim != 2 or emb.shape[0] != kept:
                    raise RuntimeError(f"Unexpected emb shape at hpo={hid}: {emb.shape}, kept={kept}")

                if inferred_D is None:
                    inferred_D = int(emb.shape[1])
                elif int(emb.shape[1]) != int(inferred_D):
                    raise RuntimeError(f"Embedding dim changed: got {emb.shape[1]} but expected {inferred_D}")

                for i in range(emb.shape[0]):
                    v = l2_normalize_f32(emb[i])
                    if not np.isfinite(v).all():
                        raise RuntimeError(f"Non-finite medium embedding at hpo={hid} row={row_idx} line_i={i}")
                    all_vecs.append(v)

                end = start + emb.shape[0]
                cursor = end
            else:
                end = start + kept
                cursor = end
        else:
            mask_med[row_idx] = 0

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

    # Always save index/mask/meta; save embeddings only if not dry_run
    write_json(out_index, med_index)
    atomic_save_npy(out_mask, mask_med.astype(np.uint8))
    write_jsonl(out_meta, meta_rows)

    # schema warning based on NOT_FOUND ratio among rows_with_hid
    rows_with_hid = int(row_counters.get("rows_with_hid", 0))
    not_found = int(key_hits.get("NOT_FOUND", 0))
    ratio_nf = float(not_found) / float(max(1, rows_with_hid))

    if args.dry_run:
        notes = [
            "dry_run does not embed; cursor is virtual.",
            "key_hits shows which field name provided lines (NOT_FOUND indicates mismatch).",
            "Outputs saved under base_out_dir/Medium_embed.",
        ]
        if ratio_nf > 0.5:
            notes.append(f"[WARN] NOT_FOUND ratio among rows_with_hid={ratio_nf:.3f} (schema mismatch likely).")

        stats = {
            "created_at": now_str(),
            "mode": "dry_run",
            "b2_path": args.b2_path,
            "master_hpo_ids": master_path,
            "base_out_dir": args.base_out_dir,
            "out_dir": out_dir,
            "N": N,
            "row_counters": row_counters,
            "key_hits": dict(key_hits),
            "NOT_FOUND_ratio_among_rows_with_hid": ratio_nf,
            "raw_lines_quantiles": quantiles_int(raw_counts),
            "kept_lines_quantiles": quantiles_int(kept_counts),
            "coverage_hpo_with_medium": int(mask_med.sum()),
            "files": {
                "med_index": out_index,
                "mask_med": out_mask,
                "meta": out_meta,
                "stats": out_stats,
            },
            "notes": notes,
        }
        write_json(out_stats, stats)

        tqdm.write("[DONE] DRY RUN saved (no embeddings):")
        tqdm.write(f"  {out_index}")
        tqdm.write(f"  {out_mask}")
        tqdm.write(f"  {out_meta}")
        tqdm.write(f"  {out_stats}")
        tqdm.write(f"  (dir) {out_dir}")
        return

    # Stack vectors
    if all_vecs:
        E = np.stack(all_vecs, axis=0).astype(np.float32)  # [L, D]
    else:
        E = np.zeros((0, int(inferred_D) if inferred_D is not None else 0), dtype=np.float32)

    # sanity: unit norm
    if E.shape[0] > 0:
        norms = np.linalg.norm(E, axis=1)
        if not np.isfinite(norms).all():
            raise RuntimeError("Non-finite norms found in E_med_lines")
        bad = int(np.sum(np.abs(norms - 1.0) > 1e-3))
        if bad > 0:
            tqdm.write(f"[WARN] {bad}/{E.shape[0]} medium vectors deviate from unit norm by > 1e-3")

    atomic_save_npy(out_lines_npy, E)

    notes = [
        "E_med_lines is a flattened pool of all medium lines across HPOs.",
        "med_index maps each hpo_id to [start,end) row range in E_med_lines.",
        "mask_med is aligned to master_hpo_ids order (same N).",
        "Outputs saved under base_out_dir/Medium_embed.",
    ]
    if ratio_nf > 0.5:
        notes.append(f"[WARN] NOT_FOUND ratio among rows_with_hid={ratio_nf:.3f} (schema mismatch likely).")

    stats = {
        "created_at": now_str(),
        "b2_path": args.b2_path,
        "master_hpo_ids": master_path,
        "base_out_dir": args.base_out_dir,
        "out_dir": out_dir,
        "N": N,
        "row_counters": row_counters,
        "key_hits": dict(key_hits),
        "NOT_FOUND_ratio_among_rows_with_hid": ratio_nf,
        "L_total_lines_embedded": int(E.shape[0]),
        "D": int(E.shape[1]) if E.ndim == 2 else 0,
        "coverage_hpo_with_medium": int(mask_med.sum()),
        "raw_lines_quantiles": quantiles_int(raw_counts),
        "kept_lines_quantiles": quantiles_int(kept_counts),
        "params": {
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "max_lines_per_hpo": int(args.max_lines_per_hpo),
            "min_line_len": int(args.min_line_len),
            "dedup": bool(args.dedup),
        },
        "files": {
            "E_med_lines": out_lines_npy,
            "med_index": out_index,
            "mask_med": out_mask,
            "meta": out_meta,
            "stats": out_stats,
        },
        "notes": notes,
    }
    write_json(out_stats, stats)

    tqdm.write("[DONE] Saved:")
    tqdm.write(f"  {out_lines_npy}")
    tqdm.write(f"  {out_index}")
    tqdm.write(f"  {out_mask}")
    tqdm.write(f"  {out_meta}")
    tqdm.write(f"  {out_stats}")
    tqdm.write(f"  (dir) {out_dir}")


if __name__ == "__main__":
    main()
