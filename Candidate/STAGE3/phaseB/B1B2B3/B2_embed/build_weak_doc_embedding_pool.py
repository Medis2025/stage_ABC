#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_weak_doc_embedding_pool.py  (SAVE TO out/weak)

Build a flattened embedding pool for PubMed "weak evidence" docs from PhaseA evidence_pool.jsonl,
aligned to the DEF master HPO id order (hpo_ids.json).

Input (example):
  /cluster/home/gw/.../STAGE3/out_phaseA/<run>/pubmed/evidence_pool.jsonl

Base out dir (default, created if missing):
  /cluster/home/gw/.../STAGE3/phaseB/B1B2B3/B2_embed/out

THIS SCRIPT writes into:
  <base_out_dir>/weak   (or other subdir via --subdir)

Outputs:
- E_weak_docs.npy     [L, D] float32, unit norm (CPU exact normalize)
- weak_index.json     {hpo_id: [start,end), ...}
- mask_weak.npy       [N] uint8 aligned to master hpo_ids.json order
- weak_source.jsonl   per HPO stats (raw/kept counts, start/end)
- weak_docs_meta.jsonl  per embedded doc record (optional but recommended)
- stats_weak.json     global stats

Alignment:
- Uses master_hpo_ids.json (default: <base_out_dir>/hpo_ids.json from DEF step).
  Typically: <base_out_dir>/Def_embed/hpo_ids.json

Dedup / selection:
- Deduplicate docs per HPO by sha1_abstract_norm (fallback: pmid).
- Optional clamp: keep up to --max_docs_per_hpo docs per HPO.
- Ranking when clamping: recent-first (year_int desc), then len_abstract_norm desc.

Embedding:
- Embeds doc text = title + "\\n" + abstract  (cleaned whitespace).
- Uses Qwen3EmbeddingClient (same as your Medium script).
- Forces unit-norm on CPU via l2_normalize_f32.

Atomic writes:
- write_json / write_jsonl use .tmp and os.replace
- atomic_save_npy uses tmp ending with .npy to avoid np.save auto-appending.

Typical runs:
1) Dry run (stats only):
   python3 build_weak_doc_embedding_pool.py --dry_run

2) Full run:
   python3 build_weak_doc_embedding_pool.py --dedup --max_docs_per_hpo 20 --batch_size 16 --max_length 512
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


# -------------------------
# defaults (match your tree)
# -------------------------

DEFAULT_BASE_OUT_DIR = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out"
DEFAULT_MODEL_DIR = "/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B"

DEFAULT_EVIDENCE_PATH = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/out_phaseA/20260126_164618/pubmed/evidence_pool.jsonl"


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


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    # non-atomic append for large meta logs; still safe enough for batch use
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    num = re.sub(r"\D", "", x)
    if len(num) == 7:
        return f"HP:{num}"
    return x if x.startswith("HP:") else ""


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
# loading
# -------------------------

def load_jsonl_stream(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def get_hpo_id(obj: Dict[str, Any]) -> str:
    for k in ["hpo_id", "Id", "id", "HPO_ID", "hp_id", "hpo", "hp", "hid", "HPO"]:
        v = obj.get(k)
        if isinstance(v, str):
            hid = normalize_hp(v)
            if hid:
                return hid
    if "HPO" in obj and isinstance(obj["HPO"], dict):
        for k in ["hpo_id", "Id", "id", "HPO_ID", "hp_id", "hpo", "hp", "hid"]:
            v = obj["HPO"].get(k)
            if isinstance(v, str):
                hid = normalize_hp(v)
                if hid:
                    return hid
    return ""


def get_str(obj: Dict[str, Any], key: str, default: str = "") -> str:
    v = obj.get(key)
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v
    return default


def get_int(obj: Dict[str, Any], key: str, default: int = -1) -> int:
    v = obj.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        vv = v.strip()
        if vv.isdigit() or (vv.startswith("-") and vv[1:].isdigit()):
            return int(vv)
    return default


def get_bool(obj: Dict[str, Any], key: str, default: bool = False) -> bool:
    v = obj.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in ("1", "true", "yes", "y"):
            return True
        if vv in ("0", "false", "no", "n"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def build_hpo_to_docs(
    evidence_path: str,
    require_abstract: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], Counter]:
    """
    Parse evidence_pool.jsonl and build:
      hpo2docs[hpo_id] = list of doc dicts with keys:
        pmid, doc_key, sha1, year_int, len_abstract_norm, title, abstract, text

    Returns:
      hpo2docs
      counters (rows_total, rows_with_hid, rows_kept_basic, rows_missing_title, rows_missing_abstract, ...)
      key_hits (observed keys for sanity)
    """
    hpo2docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    counters: Dict[str, int] = defaultdict(int)
    key_hits = Counter()

    for obj in load_jsonl_stream(evidence_path):
        counters["rows_total"] += 1
        if not isinstance(obj, dict):
            counters["rows_not_dict"] += 1
            continue

        hid = get_hpo_id(obj)
        if not hid:
            counters["rows_no_hid"] += 1
            continue
        counters["rows_with_hid"] += 1

        # observe typical keys
        for k in ["pmid", "title", "abstract", "year_int", "len_abstract_norm", "sha1_abstract_norm", "has_abstract", "journal", "year"]:
            if k in obj:
                key_hits[k] += 1

        pmid = clean_line(get_str(obj, "pmid", ""))
        title = clean_line(get_str(obj, "title", ""))
        abstract = clean_line(get_str(obj, "abstract", ""))
        has_abs = get_bool(obj, "has_abstract", default=bool(abstract))
        year_int = get_int(obj, "year_int", default=-1)
        len_abs = get_int(obj, "len_abstract_norm", default=len(abstract))
        sha1 = clean_line(get_str(obj, "sha1_abstract_norm", ""))

        if not pmid:
            counters["rows_no_pmid"] += 1
            continue
        if not title:
            counters["rows_missing_title"] += 1
        if not abstract:
            counters["rows_missing_abstract"] += 1

        if require_abstract and (not has_abs or not abstract):
            counters["rows_dropped_no_abstract"] += 1
            continue

        text = (title + "\n" + abstract).strip()
        text = clean_line(text.replace("\n", " \n ").replace(" \n ", "\n"))  # keep a single newline separator
        if not text or len(text) < 8:
            counters["rows_dropped_short_text"] += 1
            continue

        doc = {
            "hpo_id": hid,
            "pmid": pmid,
            "doc_key": f"pmid:{pmid}",
            "sha1": sha1,
            "year_int": year_int,
            "len_abstract_norm": len_abs,
            "title": title,
            "abstract": abstract,
            "text": text,
            "journal": get_str(obj, "journal", ""),
            "year": get_str(obj, "year", ""),
        }
        hpo2docs[hid].append(doc)
        counters["rows_kept_basic"] += 1

    return hpo2docs, dict(counters), key_hits


def dedup_and_select_docs(
    docs: List[Dict[str, Any]],
    dedup: bool,
    max_docs_per_hpo: int,
    min_abs_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Deduplicate docs per HPO and optionally clamp to top-K.

    Ranking for clamping:
      - year_int desc (unknown year=-1 goes last)
      - len_abstract_norm desc
      - pmid (stable)
    """
    stats: Dict[str, int] = {}
    raw = len(docs)
    stats["raw_docs"] = raw

    # filter by abstract length
    if min_abs_len > 0:
        docs = [d for d in docs if int(d.get("len_abstract_norm", 0) or 0) >= int(min_abs_len)]
    stats["after_min_abs_len"] = len(docs)

    # dedup
    if dedup and docs:
        seen = set()
        uniq = []
        for d in docs:
            key = d.get("sha1") or d.get("pmid") or d.get("doc_key")
            if not key:
                key = d.get("doc_key")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        docs = uniq
    stats["after_dedup"] = len(docs)

    # sort for clamping
    def _k(d):
        y = int(d.get("year_int", -1) or -1)
        l = int(d.get("len_abstract_norm", 0) or 0)
        return (-y if y >= 0 else 10**9, -l, str(d.get("pmid", "")))

    docs.sort(key=_k)

    # clamp
    if max_docs_per_hpo > 0 and len(docs) > max_docs_per_hpo:
        docs = docs[:max_docs_per_hpo]
    stats["kept_docs"] = len(docs)
    return docs, stats


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser("build_weak_doc_embedding_pool (save to out/weak)")
    ap.add_argument("--evidence_path", type=str, default=DEFAULT_EVIDENCE_PATH, help="PhaseA pubmed/evidence_pool.jsonl")

    ap.add_argument(
        "--base_out_dir",
        type=str,
        default=DEFAULT_BASE_OUT_DIR,
        help="Base output dir (script will create <base_out_dir>/<subdir>)",
    )
    ap.add_argument("--subdir", type=str, default="weak", help="Subdirectory under base_out_dir (default=weak)")
    ap.add_argument(
        "--master_hpo_ids",
        type=str,
        default="",
        help="Master hpo_ids.json (row order) to align mask_weak.npy. Default=<base_out_dir>/hpo_ids.json",
    )

    ap.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / cuda:0 ... (empty=auto)")
    ap.add_argument("--dtype", type=str, default="", help="bf16/fp16/fp32 (empty=auto)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--max_docs_per_hpo", type=int, default=20, help="Clamp docs per HPO (0=no clamp)")
    ap.add_argument("--min_abs_len", type=int, default=50, help="Drop docs with too-short normalized abstract length")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate docs per HPO by sha1_abstract_norm/pmid")
    ap.add_argument("--allow_missing_title", action="store_true", help="Allow empty title (still requires abstract by default)")
    ap.add_argument("--allow_missing_abstract", action="store_true", help="Allow missing abstract (NOT recommended)")
    ap.add_argument("--dry_run", action="store_true", help="Parse + stats only, do not embed")

    ap.add_argument("--write_doc_meta", action="store_true", help="Write per-doc metadata jsonl (recommended for debug/audit)")

    args = ap.parse_args()

    out_dir = os.path.join(args.base_out_dir, args.subdir)
    ensure_dir(out_dir)

    # master HPO ids
    master_path = args.master_hpo_ids.strip() or os.path.join(args.base_out_dir, "hpo_ids.json")
    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"master_hpo_ids not found: {master_path}\n"
            f"Tip: pass --master_hpo_ids <base_out_dir>/Def_embed/hpo_ids.json"
        )

    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    hpo_ids = master.get("ids", [])
    if not isinstance(hpo_ids, list) or not hpo_ids or not all(isinstance(x, str) for x in hpo_ids):
        raise ValueError("master_hpo_ids.json must be like: {\"ids\": [\"HP:...\", ...]}")

    N = len(hpo_ids)

    # outputs
    out_E = os.path.join(out_dir, "E_weak_docs.npy")
    out_index = os.path.join(out_dir, "weak_index.json")
    out_mask = os.path.join(out_dir, "mask_weak.npy")
    out_source = os.path.join(out_dir, "weak_source.jsonl")
    out_stats = os.path.join(out_dir, "stats_weak.json")
    out_doc_meta = os.path.join(out_dir, "weak_docs_meta.jsonl")  # optional

    # parse evidence pool
    require_abs = not bool(args.allow_missing_abstract)
    hpo2docs, parse_counters, key_hits = build_hpo_to_docs(args.evidence_path, require_abstract=require_abs)

    # init embedding client unless dry_run
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

    mask_weak = np.zeros((N,), dtype=np.uint8)
    weak_index: Dict[str, List[int]] = {}
    source_rows: List[Dict[str, Any]] = []
    doc_meta_rows: List[Dict[str, Any]] = []  # only if write_doc_meta and dry_run (for full run we append)

    all_vecs: List[np.ndarray] = []
    cursor = 0
    inferred_D: Optional[int] = None

    raw_counts: List[int] = []
    kept_counts: List[int] = []

    # prepare doc_meta file
    if args.write_doc_meta and (not args.dry_run):
        # truncate existing
        with open(out_doc_meta, "w", encoding="utf-8") as f:
            f.write("")

    p = tqdm(range(N), desc="WEAK align+embed", unit="hpo")
    for row_idx in p:
        hid = hpo_ids[row_idx]
        raw_docs = hpo2docs.get(hid, []) or []
        raw_counts.append(len(raw_docs))

        # optional title/abstract allowances
        if not args.allow_missing_title:
            raw_docs = [d for d in raw_docs if clean_line(str(d.get("title", "")))]

        if args.allow_missing_abstract:
            # allow missing abstract: keep doc if text exists
            raw_docs = [d for d in raw_docs if clean_line(str(d.get("text", "")))]
        else:
            raw_docs = [d for d in raw_docs if clean_line(str(d.get("abstract", "")))]

        docs_sel, sel_stats = dedup_and_select_docs(
            raw_docs,
            dedup=bool(args.dedup),
            max_docs_per_hpo=int(args.max_docs_per_hpo),
            min_abs_len=int(args.min_abs_len),
        )

        kept = int(sel_stats.get("kept_docs", 0))
        kept_counts.append(kept)

        start = cursor
        end = cursor

        if kept > 0:
            mask_weak[row_idx] = 1

            if not args.dry_run:
                assert client is not None
                texts = [d["text"] for d in docs_sel]
                emb = client.encode(texts, mode="doc", return_numpy=True)  # [m, D]
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
                        raise RuntimeError(f"Non-finite weak embedding at hpo={hid} row={row_idx} doc_i={i}")
                    all_vecs.append(v)

                    if args.write_doc_meta:
                        d = docs_sel[i]
                        append_jsonl(out_doc_meta, {
                            "hpo_id": hid,
                            "row_idx": int(row_idx),
                            "start": int(start),
                            "local_i": int(i),
                            "global_i": int(start + i),
                            "pmid": d.get("pmid", ""),
                            "doc_key": d.get("doc_key", ""),
                            "sha1": d.get("sha1", ""),
                            "year_int": int(d.get("year_int", -1) or -1),
                            "len_abstract_norm": int(d.get("len_abstract_norm", 0) or 0),
                            "journal": d.get("journal", ""),
                        })

                end = start + emb.shape[0]
                cursor = end
            else:
                # virtual cursor
                end = start + kept
                cursor = end

                if args.write_doc_meta:
                    # keep in-memory for dry_run (small)
                    for i, d in enumerate(docs_sel):
                        doc_meta_rows.append({
                            "hpo_id": hid,
                            "row_idx": int(row_idx),
                            "start": int(start),
                            "local_i": int(i),
                            "global_i": int(start + i),
                            "pmid": d.get("pmid", ""),
                            "doc_key": d.get("doc_key", ""),
                            "sha1": d.get("sha1", ""),
                            "year_int": int(d.get("year_int", -1) or -1),
                            "len_abstract_norm": int(d.get("len_abstract_norm", 0) or 0),
                            "journal": d.get("journal", ""),
                        })
        else:
            mask_weak[row_idx] = 0

        weak_index[hid] = [int(start), int(end)]

        source_rows.append({
            "hpo_id": hid,
            "row_idx": int(row_idx),
            "raw_docs": int(sel_stats.get("raw_docs", len(raw_docs))),
            "after_min_abs_len": int(sel_stats.get("after_min_abs_len", 0)),
            "after_dedup": int(sel_stats.get("after_dedup", 0)),
            "kept_docs": int(kept),
            "start": int(start),
            "end": int(end),
        })

        p.set_postfix({"L_total": cursor, "hit": int(mask_weak[row_idx])})
    p.close()

    # always save index/mask/source
    write_json(out_index, weak_index)
    atomic_save_npy(out_mask, mask_weak.astype(np.uint8))
    write_jsonl(out_source, source_rows)

    if args.write_doc_meta and args.dry_run:
        # write doc meta in dry_run too (atomic)
        write_jsonl(out_doc_meta, doc_meta_rows)

    # dry_run stats
    if args.dry_run:
        stats = {
            "created_at": now_str(),
            "mode": "dry_run",
            "evidence_path": args.evidence_path,
            "master_hpo_ids": master_path,
            "base_out_dir": args.base_out_dir,
            "out_dir": out_dir,
            "N": N,
            "parse_counters": parse_counters,
            "key_hits": dict(key_hits),
            "coverage_hpo_with_weak": int(mask_weak.sum()),
            "raw_docs_quantiles": quantiles_int(raw_counts),
            "kept_docs_quantiles": quantiles_int(kept_counts),
            "virtual_L_total_docs": int(cursor),
            "params": {
                "dedup": bool(args.dedup),
                "max_docs_per_hpo": int(args.max_docs_per_hpo),
                "min_abs_len": int(args.min_abs_len),
                "allow_missing_title": bool(args.allow_missing_title),
                "allow_missing_abstract": bool(args.allow_missing_abstract),
                "write_doc_meta": bool(args.write_doc_meta),
            },
            "files": {
                "weak_index": out_index,
                "mask_weak": out_mask,
                "weak_source": out_source,
                "weak_docs_meta": out_doc_meta if args.write_doc_meta else "",
                "stats": out_stats,
            },
            "notes": [
                "dry_run does not embed; cursor is virtual.",
                "weak_index maps each hpo_id to [start,end) range in E_weak_docs.npy.",
                "mask_weak is aligned to master_hpo_ids order (same N).",
                "Docs are title+abstract with whitespace normalized.",
                "Dedup uses sha1_abstract_norm then pmid fallback.",
                "Clamping uses recent-first (year_int desc), then len_abstract_norm desc.",
            ],
        }
        write_json(out_stats, stats)

        tqdm.write("[DONE] DRY RUN saved (no embeddings):")
        tqdm.write(f"  {out_index}")
        tqdm.write(f"  {out_mask}")
        tqdm.write(f"  {out_source}")
        if args.write_doc_meta:
            tqdm.write(f"  {out_doc_meta}")
        tqdm.write(f"  {out_stats}")
        tqdm.write(f"  (dir) {out_dir}")
        return

    # stack embeddings
    if all_vecs:
        E = np.stack(all_vecs, axis=0).astype(np.float32)  # [L, D]
    else:
        E = np.zeros((0, int(inferred_D) if inferred_D is not None else 0), dtype=np.float32)

    # sanity: unit norm
    if E.shape[0] > 0:
        norms = np.linalg.norm(E, axis=1)
        if not np.isfinite(norms).all():
            raise RuntimeError("Non-finite norms found in E_weak_docs")
        bad = int(np.sum(np.abs(norms - 1.0) > 1e-3))
        if bad > 0:
            tqdm.write(f"[WARN] {bad}/{E.shape[0]} vectors deviate from unit norm by > 1e-3")

    atomic_save_npy(out_E, E)

    stats = {
        "created_at": now_str(),
        "mode": "full",
        "evidence_path": args.evidence_path,
        "master_hpo_ids": master_path,
        "base_out_dir": args.base_out_dir,
        "out_dir": out_dir,
        "N": N,
        "parse_counters": parse_counters,
        "key_hits": dict(key_hits),
        "coverage_hpo_with_weak": int(mask_weak.sum()),
        "raw_docs_quantiles": quantiles_int(raw_counts),
        "kept_docs_quantiles": quantiles_int(kept_counts),
        "L_total_docs_embedded": int(E.shape[0]),
        "D": int(E.shape[1]) if E.ndim == 2 else 0,
        "params": {
            "dedup": bool(args.dedup),
            "max_docs_per_hpo": int(args.max_docs_per_hpo),
            "min_abs_len": int(args.min_abs_len),
            "allow_missing_title": bool(args.allow_missing_title),
            "allow_missing_abstract": bool(args.allow_missing_abstract),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "device": args.device or "auto",
            "dtype": args.dtype or "auto",
            "write_doc_meta": bool(args.write_doc_meta),
        },
        "files": {
            "E_weak_docs": out_E,
            "weak_index": out_index,
            "mask_weak": out_mask,
            "weak_source": out_source,
            "weak_docs_meta": out_doc_meta if args.write_doc_meta else "",
            "stats": out_stats,
        },
        "notes": [
            "E_weak_docs is a flattened pool of doc embeddings (title+abstract) across HPOs.",
            "weak_index maps each hpo_id to [start,end) range in E_weak_docs.",
            "mask_weak is aligned to master_hpo_ids order (same N).",
            "Dedup uses sha1_abstract_norm then pmid fallback.",
            "Clamping uses recent-first (year_int desc), then len_abstract_norm desc.",
        ],
    }
    write_json(out_stats, stats)

    tqdm.write("[DONE] Saved:")
    tqdm.write(f"  {out_E}")
    tqdm.write(f"  {out_index}")
    tqdm.write(f"  {out_mask}")
    tqdm.write(f"  {out_source}")
    if args.write_doc_meta:
        tqdm.write(f"  {out_doc_meta}")
    tqdm.write(f"  {out_stats}")
    tqdm.write(f"  (dir) {out_dir}")


if __name__ == "__main__":
    main()


"""
python3 /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/build_weak_doc_embedding_pool.py \
  --evidence_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/out_phaseA/20260126_164618/pubmed/evidence_pool.jsonl \
  --base_out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out \
  --subdir weak \
  --master_hpo_ids /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out/Def_embed/hpo_ids.json \
  --dedup \
  --max_docs_per_hpo 20 \
  --batch_size 32 \
  --max_length 512


"""