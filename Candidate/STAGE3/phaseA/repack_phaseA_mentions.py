#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repack PhaseA mentions_candidates.jsonl
--------------------------------------

- DO NOT touch original PhaseA code
- DO NOT modify any other JSONs
- Read:
    phaseA/merged/mentions_candidates.jsonl
- Write:
    phaseA/normalized/mentions_candidates.norm.jsonl

Purpose:
- provide clean, stable span-level input for B1/B2/B3
"""

import os
import json
import hashlib
import argparse
import re
from typing import Dict, Any, Iterable

from tqdm import tqdm


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_label_options(x):
    if not isinstance(x, list):
        return []
    out = []
    seen = set()
    for v in x:
        v2 = str(v).strip().upper()
        if v2 and v2 not in seen:
            seen.add(v2)
            out.append(v2)
    return out


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser("repack_phaseA_mentions")
    ap.add_argument("--phaseA_run_dir", required=True)
    args = ap.parse_args()

    run_dir = args.phaseA_run_dir.rstrip("/")

    in_path = os.path.join(run_dir, "phaseA", "merged", "mentions_candidates.jsonl")
    out_path = os.path.join(run_dir, "phaseA", "normalized", "mentions_candidates.norm.jsonl")

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    ensure_dir(os.path.dirname(out_path))

    # ---------- tqdm setup ----------
    total = count_lines(in_path)

    n_in = 0
    n_out = 0

    with open(out_path + ".tmp", "w", encoding="utf-8") as fo:
        for rec in tqdm(
            iter_jsonl(in_path),
            total=total,
            desc="Repacking PhaseA mentions",
            unit="span",
            ncols=100,
        ):
            n_in += 1

            pmcid = str(rec.get("pmcid") or "")
            chunk_key = str(rec.get("chunk_key") or "")
            surface = str(rec.get("surface") or "")
            unit = str(rec.get("unit") or "")

            span_id = sha1(f"{pmcid}|{chunk_key}|{surface}|{unit}")

            out = dict(rec)  # shallow copy

            out.update({
                "span_id": span_id,
                "surface_norm": compact_ws(surface),
                "unit_norm": unit.strip().lower(),
                "context_norm": compact_ws(rec.get("context")),
                "label_options_norm": normalize_label_options(rec.get("label_options")),
                "source_stage": "PhaseA",
            })

            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    os.replace(out_path + ".tmp", out_path)

    print(f"[DONE] repacked PhaseA mentions")
    print(f"  input : {in_path}")
    print(f"  output: {out_path}")
    print(f"  rows  : {n_in} -> {n_out}")


if __name__ == "__main__":
    main()
