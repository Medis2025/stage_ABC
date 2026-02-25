#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stress_test_term_sampler.py  (REVISED: multiprocessing-safe + LLM counters + HPO fail rate)

Purpose
- Pressure test ClusterTermSampler output correctness:
  - actually returns TermRecord objects
  - optionally checks esearch hits
  - optionally triggers LLM repair (if enabled)
- Summarize stats + write jsonl outputs for inspection

Fixes / Adds
- ✅ Remove lambda inside multiprocessing (pickle-safe)
- ✅ Top-level packer function for imap_unordered
- ✅ Optional mp start method (spawn|fork|forkserver)
- ✅ LLM call counters (calls / ok / fail) aggregated across processes
- ✅ Prints LLM counters to terminal + writes into summary.json
- ✅ HPO-level metrics:
    - hpo_ok:    seed_hpo where at least 1 query hits (used_hits_pos >= 1)
    - hpo_fail:  seed_hpo where all queries are zero-hit (used_hits_pos == 0)
    - prints + writes into summary.json

Notes
- We DO NOT modify cluster_term_sampler.py
- We wrap LLMClient with CountingLLM that intercepts llm.run()
"""

from __future__ import annotations

import os
import json
import time
import argparse
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm

from cluster_term_sampler import ClusterTermSampler, TermRecord  # type: ignore
from pubmed_pmc_client import PubMedPMCClient, NCBIConfig  # type: ignore

try:
    from llm_client import LLMClient  # type: ignore
except Exception:
    LLMClient = None


# -----------------------------
# IO
# -----------------------------

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _rec_to_json(r: TermRecord) -> Dict[str, Any]:
    return {
        "kind": r.kind,
        "seed_hpo": r.seed_hpo,
        "neighbor_hpos": r.neighbor_hpos,
        "phrases": r.phrases,
        "query": r.query,
        "hits": r.hits,
        "repaired": r.repaired,
        "repaired_phrases": r.repaired_phrases,
        "repaired_query": r.repaired_query,
        "repaired_hits": r.repaired_hits,
        "used_query": r.used_query,
        "used_hits": r.used_hits,
    }


def _load_seeds_from_queries_jsonl(queries_jsonl: str, max_seeds: int = 0) -> List[str]:
    seeds: List[str] = []
    with open(queries_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            hid = (obj.get("hpo_id") or "").strip()
            if hid:
                seeds.append(hid)
            if max_seeds and len(seeds) >= max_seeds:
                break
    # unique preserve order
    return list(dict.fromkeys(seeds))


# -----------------------------
# Counting LLM wrapper
# -----------------------------

class CountingLLM:
    """
    Wrap an underlying LLMClient and count:
      - calls: number of llm.run invocations
      - ok: number of calls that returned non-empty string (best-effort)
      - fail: number of calls that raised or returned empty
    """
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.ok = 0
        self.fail = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        try:
            out = self.inner.run(*args, **kwargs)
            if isinstance(out, str) and out.strip():
                self.ok += 1
            else:
                self.fail += 1
            return out
        except Exception:
            self.fail += 1
            raise


def _build_llm(api_key_env: str, base_url: str, model: str, timeout: float) -> Optional["CountingLLM"]:
    if LLMClient is None:
        return None
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        return None
    inner = LLMClient(api_key=api_key, base_url=base_url, model=model, timeout=timeout)
    return CountingLLM(inner)


def _build_pubmed_client(email: str, api_key: str, tool: str, polite_sleep: float) -> PubMedPMCClient:
    cfg = NCBIConfig(
        email=email,
        tool=tool,
        api_key=(api_key or None),
        polite_sleep=float(polite_sleep),
    )
    return PubMedPMCClient(cfg)


# -----------------------------
# Worker
# -----------------------------

def _run_one_seed(
    seed_hpo: str,
    queries_jsonl: str,
    neighbors_jsonl: str,
    seed: int,
    n_seed0: int,
    n_seed1: int,
    n_seed2: int,
    n_neg: int,
    check_esearch: bool,
    retmax_for_check: int,
    use_llm_repair: bool,
    llm_api_key_env: str,
    llm_base_url: str,
    llm_model: str,
    llm_timeout: float,
    email: str,
    ncbi_api_key: str,
    tool: str,
    polite_sleep: float,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
    """
    Returns:
      (seed_hpo, records_json, stats_dict, llm_counts_dict)
    """

    sampler = ClusterTermSampler(
        queries_jsonl=queries_jsonl,
        neighbors_jsonl=neighbors_jsonl,
        seed=seed,
        use_case_filter=True,
    )

    pubmed_client = _build_pubmed_client(email, ncbi_api_key, tool, polite_sleep) if check_esearch else None
    llm = _build_llm(llm_api_key_env, llm_base_url, llm_model, llm_timeout) if use_llm_repair else None

    recs: List[TermRecord] = []

    for _ in range(max(0, n_seed0)):
        r = sampler.build_seed0(seed_hpo)
        if r:
            recs.append(
                sampler.maybe_repair(
                    r,
                    llm=llm,
                    client=pubmed_client,
                    check_esearch=check_esearch,
                    retmax_for_check=retmax_for_check,
                )
            )

    for _ in range(max(0, n_seed1)):
        r = sampler.build_seed1(seed_hpo)
        if r:
            recs.append(
                sampler.maybe_repair(
                    r,
                    llm=llm,
                    client=pubmed_client,
                    check_esearch=check_esearch,
                    retmax_for_check=retmax_for_check,
                )
            )

    for _ in range(max(0, n_seed2)):
        r = sampler.build_seed2(seed_hpo)
        if r:
            recs.append(
                sampler.maybe_repair(
                    r,
                    llm=llm,
                    client=pubmed_client,
                    check_esearch=check_esearch,
                    retmax_for_check=retmax_for_check,
                )
            )

    for _ in range(max(0, n_neg)):
        r = sampler.build_neg(seed_hpo)
        if r:
            recs.append(
                sampler.maybe_repair(
                    r,
                    llm=llm,
                    client=pubmed_client,
                    check_esearch=check_esearch,
                    retmax_for_check=retmax_for_check,
                )
            )

    rows = [_rec_to_json(r) for r in recs]

    repaired = sum(1 for r in recs if r.repaired)
    total = len(recs)
    hits0 = sum(1 for r in recs if (r.used_hits == 0))
    has_hits = sum(1 for r in recs if (isinstance(r.used_hits, int) and r.used_hits > 0))
    none_hits = sum(1 for r in recs if (r.used_hits is None))

    stats = {
        "seed_hpo": seed_hpo,
        "n_records": total,
        "n_repaired": repaired,
        "repair_rate": (repaired / total) if total else 0.0,
        "used_hits_zero": hits0,
        "used_hits_pos": has_hits,
        "used_hits_none": none_hits,
    }

    llm_counts = {
        "calls": int(getattr(llm, "calls", 0) or 0) if llm is not None else 0,
        "ok": int(getattr(llm, "ok", 0) or 0) if llm is not None else 0,
        "fail": int(getattr(llm, "fail", 0) or 0) if llm is not None else 0,
    }

    return seed_hpo, rows, stats, llm_counts


# ---- multiprocessing-safe wrapper (TOP-LEVEL) ----
def _run_one_seed_pack(job_args):
    return _run_one_seed(*job_args)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser("stress_test_term_sampler")
    ap.add_argument("--queries_jsonl", type=str, required=True)
    ap.add_argument("--neighbors_jsonl", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="./_stress_out")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_seeds", type=int, default=50, help="0 means all seeds in queries_jsonl")
    ap.add_argument("--n_seed0", type=int, default=1)
    ap.add_argument("--n_seed1", type=int, default=1)
    ap.add_argument("--n_seed2", type=int, default=1)
    ap.add_argument("--n_neg", type=int, default=1)

    ap.add_argument("--check_esearch", action="store_true")
    ap.add_argument("--retmax_for_check", type=int, default=1, help="IMPORTANT: set >=1 to avoid always-0 fallback")

    # NCBI
    ap.add_argument("--email", type=str, default="")
    ap.add_argument("--ncbi_api_key", type=str, default="")
    ap.add_argument("--tool", type=str, default="hpo-term-sampler-stress")
    ap.add_argument("--polite_sleep", type=float, default=0.34)

    # LLM repair
    ap.add_argument("--use_llm_repair", action="store_true")
    ap.add_argument("--llm_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--llm_model", type=str, default="deepseek-chat")
    ap.add_argument("--llm_timeout", type=float, default=60.0)

    # pressure
    ap.add_argument("--workers", type=int, default=1, help="processes. use >1 for pressure test")
    ap.add_argument("--mp_start", type=str, default="", help="optional: spawn|fork|forkserver (empty=default)")

    args = ap.parse_args()

    # Optional: force a start method (sometimes safer on HPC)
    if args.mp_start.strip():
        try:
            mp.set_start_method(args.mp_start.strip(), force=True)
        except RuntimeError:
            pass

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    seeds = _load_seeds_from_queries_jsonl(
        args.queries_jsonl,
        max_seeds=int(args.max_seeds) if args.max_seeds else 0
    )
    if not seeds:
        raise RuntimeError("No seeds (hpo_id) found in queries_jsonl")

    print(f"[INFO] loaded seeds: {len(seeds)}; first5={seeds[:5]}")
    print(f"[INFO] use_llm_repair={bool(args.use_llm_repair)}  check_esearch={bool(args.check_esearch)}  workers={int(args.workers)}")

    jobs = []
    for s in seeds:
        jobs.append((
            s,
            args.queries_jsonl,
            args.neighbors_jsonl,
            int(args.seed),
            int(args.n_seed0),
            int(args.n_seed1),
            int(args.n_seed2),
            int(args.n_neg),
            bool(args.check_esearch),
            int(args.retmax_for_check),
            bool(args.use_llm_repair),
            str(args.llm_api_key_env),
            str(args.llm_base_url),
            str(args.llm_model),
            float(args.llm_timeout),
            str(args.email),
            str(args.ncbi_api_key),
            str(args.tool),
            float(args.polite_sleep),
        ))

    all_stats: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    # LLM counters aggregated across all seeds (and processes)
    llm_calls_total = 0
    llm_ok_total = 0
    llm_fail_total = 0

    if int(args.workers) <= 1:
        for j in tqdm(jobs, desc="stress seeds", dynamic_ncols=True):
            seed_hpo, rows, st, llm_ct = _run_one_seed(*j)
            all_rows.extend(rows)
            all_stats.append(st)
            llm_calls_total += llm_ct.get("calls", 0)
            llm_ok_total += llm_ct.get("ok", 0)
            llm_fail_total += llm_ct.get("fail", 0)
    else:
        workers = int(args.workers)
        with mp.Pool(processes=workers) as pool:
            for seed_hpo, rows, st, llm_ct in tqdm(
                pool.imap_unordered(_run_one_seed_pack, jobs),
                total=len(jobs),
                desc=f"stress seeds x{workers}",
                dynamic_ncols=True
            ):
                all_rows.extend(rows)
                all_stats.append(st)
                llm_calls_total += llm_ct.get("calls", 0)
                llm_ok_total += llm_ct.get("ok", 0)
                llm_fail_total += llm_ct.get("fail", 0)

    _write_jsonl(os.path.join(run_dir, "terms_stress.jsonl"), all_rows)
    _write_jsonl(os.path.join(run_dir, "stats_per_seed.jsonl"), all_stats)

    # -----------------------------
    # Query-level summary
    # -----------------------------
    total = sum(x["n_records"] for x in all_stats)
    repaired = sum(x["n_repaired"] for x in all_stats)
    hits0 = sum(x["used_hits_zero"] for x in all_stats)
    hitspos = sum(x["used_hits_pos"] for x in all_stats)

    # -----------------------------
    # HPO-level success / failure
    # HPO OK  = at least 1 query hits (used_hits_pos >= 1)
    # HPO FAIL= all queries zero-hit (used_hits_pos == 0)
    # -----------------------------
    hpo_total = len(all_stats)
    hpo_ok = 0
    hpo_fail = 0
    for st in all_stats:
        if st.get("used_hits_pos", 0) >= 1:
            hpo_ok += 1
        else:
            hpo_fail += 1

    hpo_ok_rate = (hpo_ok / hpo_total) if hpo_total else 0.0
    hpo_fail_rate = (hpo_fail / hpo_total) if hpo_total else 0.0

    summary = {
        "run_dir": run_dir,
        "n_seeds": len(all_stats),
        "n_records_total": total,
        "n_repaired_total": repaired,
        "repair_rate_total": (repaired / total) if total else 0.0,
        "used_hits_zero_total": hits0,
        "used_hits_pos_total": hitspos,
        "check_esearch": bool(args.check_esearch),
        "retmax_for_check": int(args.retmax_for_check),
        "use_llm_repair": bool(args.use_llm_repair),
        "workers": int(args.workers),
        "llm_counts_total": {
            "calls": llm_calls_total,
            "ok": llm_ok_total,
            "fail": llm_fail_total,
        },
        "hpo_level": {
            "hpo_total": hpo_total,
            "hpo_ok": hpo_ok,
            "hpo_fail": hpo_fail,
            "hpo_ok_rate": hpo_ok_rate,
            "hpo_fail_rate": hpo_fail_rate,
        },
        "paths": {
            "terms_stress": os.path.join(run_dir, "terms_stress.jsonl"),
            "stats_per_seed": os.path.join(run_dir, "stats_per_seed.jsonl"),
        }
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- PRINT TO TERMINAL ----
    print("\n========== [LLM COUNTS] ==========")
    print(f"llm_calls_total = {llm_calls_total}")
    print(f"llm_ok_total    = {llm_ok_total}")
    print(f"llm_fail_total  = {llm_fail_total}")
    if llm_calls_total > 0:
        ok_rate = llm_ok_total / llm_calls_total
        fail_rate = llm_fail_total / llm_calls_total
        print(f"llm_ok_rate     = {ok_rate:.4f}")
        print(f"llm_fail_rate   = {fail_rate:.4f}")
    else:
        print("llm_ok_rate     = N/A (no calls)")
        print("llm_fail_rate   = N/A (no calls)")
    print("==================================\n")

    print("\n========== [HPO-LEVEL METRICS] ==========")
    print(f"hpo_total      = {hpo_total}")
    print(f"hpo_ok         = {hpo_ok}")
    print(f"hpo_fail       = {hpo_fail}")
    print(f"hpo_ok_rate    = {hpo_ok_rate:.4f}")
    print(f"hpo_fail_rate  = {hpo_fail_rate:.4f}")
    print("========================================\n")

    print("\n[DONE] stress test finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
