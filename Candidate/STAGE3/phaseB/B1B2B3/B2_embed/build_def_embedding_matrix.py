#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_def_embedding_matrix.py

Embed (DEF strong prior) for ALL HPO IDs in:
  /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json

Strategy (as agreed):
- Def is list -> join into a paragraph
- llm_def (if exists) > llm_add_def
- Comment is appended ONLY when curated Def is short (optional gate)
- Always return non-empty text (fallback to Name)

Outputs (saved under):
  .../out/Def_embed

Files:
- E_def.npy                  [N, D] float32, L2-normalized (exact CPU normalize)
- hpo_ids.json               {"ids": [...]} in the exact row order of E_def.npy
- def_source.jsonl           one line per HPO: hpo_id, name, def_source, def_len
- stats.json                 summary counts + length quantiles

Notes:
- Uses Qwen3EmbeddingClient from qwen_clients.py
- Uses tqdm progress bars
- Creates output directory if missing
"""

from __future__ import annotations

import os
import json
import argparse
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from tqdm import tqdm

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


DEFAULT_HPO_JSON = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json"
DEFAULT_MODEL_DIR = "/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B"

# NOTE: minimized change: default out dir now points to .../out/Def_embed
DEFAULT_OUT_DIR = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out/Def_embed"


# -------------------------
# small utils
# -------------------------

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

def join_list(xs) -> str:
    if xs is None:
        return ""
    if isinstance(xs, str):
        xs = [xs]
    if not isinstance(xs, list):
        xs = [str(xs)]
    parts = []
    for x in xs:
        t = str(x).strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()

def l2_normalize_f32(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0:
        return np.zeros_like(v, dtype=np.float32)
    return (v / (n + 1e-12)).astype(np.float32)

def quantiles_int(xs: List[int], qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    arr = np.array(xs, dtype=np.float32)
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(arr, q))
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["mean"] = float(arr.mean())
    return out

def chunk_list(xs: List[Any], bs: int) -> List[List[Any]]:
    bs = max(1, int(bs))
    return [xs[i:i + bs] for i in range(0, len(xs), bs)]


# -------------------------
# atomic npy save (FIX)
# -------------------------

def atomic_save_npy(path_npy: str, arr: np.ndarray) -> None:
    """
    Atomic save for .npy without numpy auto-appending ".npy".
    Fixes: np.save('xxx.tmp') => 'xxx.tmp.npy' pitfall.
    """
    ensure_dir(os.path.dirname(path_npy))
    tmp = path_npy + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path_npy)


# -------------------------
# def selection strategy
# -------------------------

def pick_def_text(
    obj: dict,
    *,
    min_len: int = 60,
    use_comment: bool = True,
    comment_gate_len: int = 120,
) -> Tuple[str, str, str]:
    """
    Returns: (def_text, source, name)
    Strategy:
    - curated Def is list => join
    - llm_def > llm_add_def
    - comment appended ONLY when curated_def is short (gate)
    - fallback to name if everything else empty
    """
    # tolerate different key styles
    name = join_list(obj.get("Name") or obj.get("name") or [])
    curated = join_list(obj.get("Def") or obj.get("def") or [])
    comment = join_list(obj.get("Comment") or obj.get("comment") or [])
    llm_def = str(obj.get("llm_def") or "").strip()
    llm_add = str(obj.get("llm_add_def") or "").strip()

    # whether curated is short
    short_curated = (len(curated) < min_len) and (len(curated) > 0)

    # base selection
    base = curated
    source = "curated_def"
    if len(base) < min_len:
        if llm_def:
            base, source = llm_def, "llm_def"
        elif llm_add:
            base, source = llm_add, "llm_add_def"
        elif name:
            base, source = name, "name_fallback"
        else:
            base, source = "[EMPTY]", "empty_fallback"

    # comment augmentation only when we are still using curated_def
    if (
        use_comment
        and source == "curated_def"
        and short_curated
        and comment
        and len(curated) < comment_gate_len
    ):
        base = curated.rstrip(".") + ". " + comment
        source = "curated_def+comment"

    # ensure non-empty
    if not base.strip():
        base = name.strip() or "[EMPTY]"
        source = "name_fallback" if name.strip() else "empty_fallback"

    return base, source, name


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser("build_def_embedding_matrix")
    ap.add_argument("--hpo_json", type=str, default=DEFAULT_HPO_JSON)
    ap.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    ap.add_argument("--device", type=str, default="", help="cuda / cpu / cuda:0 ... (empty=auto)")
    ap.add_argument("--dtype", type=str, default="", help="bf16/fp16/fp32 (empty=auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--min_len", type=int, default=60, help="curated def length threshold")
    ap.add_argument("--use_comment", action="store_true", help="enable curated short-def + comment augmentation")
    ap.add_argument("--comment_gate_len", type=int, default=120)

    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N HPOs (debug)")
    ap.add_argument("--seed", type=int, default=0, help="If >0, shuffle HPO order with this seed (debug). Default keeps sorted order.")

    args = ap.parse_args()

    if not os.path.exists(args.hpo_json):
        raise FileNotFoundError(f"hpo_json not found: {args.hpo_json}")

    ensure_dir(args.out_dir)

    # Load HPO json
    with open(args.hpo_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all HP:* keys and make deterministic order (sorted)
    hpo_ids = [k for k in data.keys() if isinstance(k, str) and k.startswith("HP:")]
    if not hpo_ids:
        raise RuntimeError("No HP:* keys found in hpo_json")

    hpo_ids = sorted(hpo_ids)
    if int(args.seed) > 0:
        rng = np.random.RandomState(int(args.seed))
        rng.shuffle(hpo_ids)

    if int(args.limit) > 0:
        hpo_ids = hpo_ids[: int(args.limit)]

    # Init embedding client
    client = Qwen3EmbeddingClient(EmbeddingConfig(
        model_dir=args.model_dir,
        device=(args.device or None),
        dtype=(args.dtype or None),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        normalize=True,          # we'll still do exact CPU normalize after
        use_instruct=False,      # def/doc => no instruction formatting
        attn_implementation=None,
    ))

    # Pre-build all texts (so we can save source stats even if embedding fails later)
    texts: List[str] = []
    sources: List[str] = []
    names: List[str] = []
    def_lens: List[int] = []

    p0 = tqdm(hpo_ids, desc="Prepare def_text", unit="hpo")
    for hid in p0:
        obj = data.get(hid, {}) or {}
        text, source, name = pick_def_text(
            obj,
            min_len=int(args.min_len),
            use_comment=bool(args.use_comment),
            comment_gate_len=int(args.comment_gate_len),
        )
        texts.append(text)
        sources.append(source)
        names.append(name)
        def_lens.append(len(text))
    p0.close()

    N = len(hpo_ids)

    # Embed in batches
    E: Optional[np.ndarray] = None
    rows_meta: List[Dict[str, Any]] = []

    batches = chunk_list(list(range(N)), int(args.batch_size))
    p1 = tqdm(batches, desc="Embed def_text", unit="batch")

    for idxs in p1:
        batch_texts = [texts[i] for i in idxs]
        emb = client.encode(batch_texts, mode="doc", return_numpy=True)  # [B, D]
        emb = np.asarray(emb, dtype=np.float32)

        # exact normalize + finite check
        for j, global_i in enumerate(idxs):
            v = l2_normalize_f32(emb[j])
            if not np.isfinite(v).all():
                raise RuntimeError(f"Non-finite embedding at hpo={hpo_ids[global_i]} index={global_i}")
            emb[j] = v

        if E is None:
            E = np.zeros((N, emb.shape[1]), dtype=np.float32)
        E[idxs, :] = emb

        # metadata rows
        for global_i in idxs:
            rows_meta.append({
                "hpo_id": hpo_ids[global_i],
                "name": names[global_i],
                "def_source": sources[global_i],
                "def_len": def_lens[global_i],
            })

    p1.close()

    assert E is not None, "Embedding matrix E not built"

    # Quick sanity
    norms = np.linalg.norm(E, axis=1)
    if not np.isfinite(norms).all():
        raise RuntimeError("Found non-finite norms in E_def.npy")
    bad = int(np.sum(np.abs(norms - 1.0) > 1e-3))
    if bad > 0:
        tqdm.write(f"[WARN] {bad}/{N} vectors deviate from unit norm by > 1e-3 (should be rare)")

    # Write outputs
    out_npy = os.path.join(args.out_dir, "E_def.npy")
    out_ids = os.path.join(args.out_dir, "hpo_ids.json")
    out_meta = os.path.join(args.out_dir, "def_source.jsonl")
    out_stats = os.path.join(args.out_dir, "stats.json")

    # atomic save npy (FIXED)
    atomic_save_npy(out_npy, E.astype(np.float32))

    write_json(out_ids, {"ids": hpo_ids})
    write_jsonl(out_meta, rows_meta)

    src_cnt = Counter(sources)
    stats = {
        "created_at": now_str(),
        "hpo_json": args.hpo_json,
        "model_dir": args.model_dir,
        "out_dir": args.out_dir,
        "N": N,
        "D": int(E.shape[1]),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "min_len": int(args.min_len),
        "use_comment": bool(args.use_comment),
        "comment_gate_len": int(args.comment_gate_len),
        "def_source_counts": dict(src_cnt),
        "def_len_quantiles": quantiles_int(def_lens),
        "unit_norm_bad_count_gt_1e-3": bad,
        "files": {
            "E_def": out_npy,
            "hpo_ids": out_ids,
            "def_source": out_meta,
            "stats": out_stats,
        },
        "notes": [
            "E_def.npy is float32 and CPU-renormalized to unit length.",
            "Row order is hpo_ids.json['ids'] (sorted HP:* by default).",
            "def_source.jsonl aligns 1:1 with E_def rows (same order).",
        ],
    }
    write_json(out_stats, stats)

    tqdm.write("[DONE] Saved:")
    tqdm.write(f"  {out_npy}")
    tqdm.write(f"  {out_ids}")
    tqdm.write(f"  {out_meta}")
    tqdm.write(f"  {out_stats}")


if __name__ == "__main__":
    main()
