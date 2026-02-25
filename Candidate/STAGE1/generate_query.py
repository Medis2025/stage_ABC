#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_query.py  (FULL RUN, MULTI-THREAD + RETRY + PROGRESS)

Goal:
- Load HPO JSON: data/hpo_enriched_with_llm.json
- For ALL HPO terms (or optionally a slice), call LLM with 3 stage1 prompts:
    - Descriptive_candidates.txt  -> scale2
    - Mechanism.txt              -> scale3
    - Domain.tzt                 -> scale4  (file name kept as you have it)
- Assemble a query JSON object (one per HPO) and dump to:
    Candidate/queries.jsonl   (jsonl)

Key upgrades vs TEST:
- Multi-thread generation (ThreadPoolExecutor)
- Robust retry on HTTP / transient errors per HPO + per stage
- Never crash the whole run because of one HPO term
- Progress bars for:
    (1) loading / indexing
    (2) generation over all HPO terms
    (3) writing output

Notes:
- LLM outputs plain text lines (one phrase per line), parsed into lists.
- We keep raw outputs (or error strings) in "raw" for debugging.
- By default we DO NOT cache; add caching later if desired.
"""

import os
import sys
import json
import time
import argparse
import threading
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

# Progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback to no progress bar

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APIS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "APIs"))

if APIS_DIR not in sys.path:
    sys.path.insert(0, APIS_DIR)

from llm_client import LLMClient  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers: HPO field extraction (matches your JSON style)
# -----------------------------------------------------------------------------

def _extract_id(term: Dict[str, Any]) -> str:
    return term.get("Id") or term.get("id") or term.get("hpo_id") or ""


def _extract_name(term: Dict[str, Any]) -> str:
    name = term.get("Name") or term.get("name") or term.get("label") or ""
    if isinstance(name, list):
        name = name[0] if name else ""
    return str(name).strip()


def _extract_synonyms(term: Dict[str, Any], max_n: int = 8) -> List[str]:
    syn = term.get("Synonym") or term.get("synonym") or term.get("synonyms") or []
    if isinstance(syn, str):
        syn = [syn]
    if not isinstance(syn, list):
        syn = [str(syn)]
    syn = [str(x).strip() for x in syn if str(x).strip()]
    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for s in syn:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_n:
            break
    return out


def _extract_def(term: Dict[str, Any]) -> str:
    d = term.get("Def") or term.get("def") or term.get("definition") or ""
    if isinstance(d, list):
        d = d[0] if d else ""
    return str(d).strip()


def _extract_llm_add_def(term: Dict[str, Any]) -> str:
    return str(term.get("llm_add_def") or "").strip()


def _extract_ancestor_ids(term: Dict[str, Any], max_n: int = 10) -> List[str]:
    """
    Use Father keys if present; else Is_a list.
    """
    father = term.get("Father")
    ids: List[str] = []
    if isinstance(father, dict):
        ids = [k for k, v in father.items() if v]
    else:
        isa = term.get("Is_a") or term.get("is_a") or []
        if isinstance(isa, str):
            ids = [isa]
        elif isinstance(isa, list):
            ids = [str(x) for x in isa]
    ids = [x.strip() for x in ids if str(x).strip()]
    return ids[:max_n]


def build_id_to_name(hpo_data: Any) -> Dict[str, str]:
    """
    Build global map HP:xxxx -> name (Name[0]).
    Supports dict keyed by HP id, or list.
    """
    m: Dict[str, str] = {}
    if isinstance(hpo_data, dict):
        it = hpo_data.values()
    elif isinstance(hpo_data, list):
        it = hpo_data
    else:
        return m

    for term in it:
        if not isinstance(term, dict):
            continue
        hid = _extract_id(term)
        if not hid:
            continue
        nm = _extract_name(term)
        if nm:
            m[hid] = nm
    return m


# -----------------------------------------------------------------------------
# Helpers: prompt loading / filling / parse output lines
# -----------------------------------------------------------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fill_template(template: str, **kwargs) -> str:
    """
    Very simple format replacement using {var} placeholders.
    Missing keys become "".
    """
    class SafeDict(dict):
        def __missing__(self, key):
            return ""
    return template.format_map(SafeDict(**kwargs))


def parse_lines(text: str, max_lines: int = 12) -> List[str]:
    """
    Parse plain-text LLM output (one phrase per line).
    Minimal cleanup only (strip, remove empty, de-dup preserve order).
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    out: List[str] = []
    seen = set()
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
        if len(out) >= max_lines:
            break
    return out


# -----------------------------------------------------------------------------
# Robust LLM call with retry (per stage)
# -----------------------------------------------------------------------------

TRANSIENT_HINTS = (
    "http", "https",
    "timeout", "timed out",
    "connection", "connection reset", "connection aborted",
    "502", "503", "504",
    "429", "rate limit",
    "too many requests",
    "server error",
)

def _is_transient_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(h in msg for h in TRANSIENT_HINTS)


def llm_run_with_retry(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int = 5,
    base_sleep: float = 1.0,
    max_sleep: float = 20.0,
) -> Tuple[str, Optional[str]]:
    """
    Returns: (raw_text, err_str)
      - raw_text: "" if failed
      - err_str: None if success else error message after final retry
    """
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.run(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
            return raw, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # Retry on transient; for non-transient, we still retry a little because
            # some clients wrap HTTP errors generically.
            transient = _is_transient_error(e)
            if attempt >= max_retries:
                return "", last_err

            # backoff with jitter (simple)
            sleep_s = min(max_sleep, base_sleep * (2 ** attempt))
            # tiny deterministic jitter without importing random (thread-safe)
            sleep_s = sleep_s + (0.1 * ((attempt % 3) + 1))
            time.sleep(sleep_s)
            continue

    return "", last_err


# -----------------------------------------------------------------------------
# Thread-local LLM client (safer than sharing one instance across threads)
# -----------------------------------------------------------------------------

_THREAD_LOCAL = threading.local()

def get_thread_llm(api_key: str, base_url: str, model: str, timeout: float) -> LLMClient:
    llm = getattr(_THREAD_LOCAL, "llm", None)
    if llm is None:
        llm = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        _THREAD_LOCAL.llm = llm
    return llm


# -----------------------------------------------------------------------------
# Core: generate candidates for one HPO term (safe, never raises)
# -----------------------------------------------------------------------------

def generate_for_term_safe(
    term: Dict[str, Any],
    id2name: Dict[str, str],
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    system_prompt: str,
    prompt_desc: str,
    prompt_mech: str,
    prompt_domain: str,
    temperature: float,
    stage_max_retries: int,
) -> Dict[str, Any]:
    """
    Never throws. If one stage fails, it returns empty list for that stage and records error in raw.
    """
    hid = _extract_id(term)
    name = _extract_name(term)
    syns = _extract_synonyms(term)
    definition = _extract_def(term)
    llm_add_def = _extract_llm_add_def(term)

    anc_ids = _extract_ancestor_ids(term)
    anc_names = [id2name.get(x, x) for x in anc_ids]  # fallback to id if name missing
    anc_names_block = "\n".join(anc_names)

    syn_block = "\n".join(syns) if syns else ""
    def_block = definition if definition else ""
    add_def_block = llm_add_def if llm_add_def else ""

    llm = get_thread_llm(api_key=api_key, base_url=base_url, model=model, timeout=timeout)

    # ---- Descriptive (scale2)
    user_desc = fill_template(
        prompt_desc,
        hpo_id=hid,
        name=name,
        synonyms_block=syn_block,
        def_block=def_block,
    )
    raw_desc, err_desc = llm_run_with_retry(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_desc,
        temperature=temperature,
        max_retries=stage_max_retries,
    )
    scale2 = parse_lines(raw_desc) if raw_desc else []

    # ---- Mechanism (scale3)
    user_mech = fill_template(
        prompt_mech,
        hpo_id=hid,
        name=name,
        def_block=def_block,
        llm_add_def_block=add_def_block,
    )
    raw_mech, err_mech = llm_run_with_retry(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_mech,
        temperature=temperature,
        max_retries=stage_max_retries,
    )
    scale3 = parse_lines(raw_mech) if raw_mech else []

    # ---- Domain/System (scale4)
    user_dom = fill_template(
        prompt_domain,
        hpo_id=hid,
        name=name,
        ancestor_names_block=anc_names_block,
    )
    raw_dom, err_dom = llm_run_with_retry(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_dom,
        temperature=temperature,
        max_retries=stage_max_retries,
    )
    scale4 = parse_lines(raw_dom) if raw_dom else []

    # Keep error strings in raw for debugging, without breaking JSON schema
    raw_block = {
        "scale2": raw_desc if raw_desc else "",
        "scale3": raw_mech if raw_mech else "",
        "scale4": raw_dom if raw_dom else "",
    }
    err_block = {
        "scale2_err": err_desc or "",
        "scale3_err": err_mech or "",
        "scale4_err": err_dom or "",
    }

    return {
        "hpo_id": hid,
        "name": name,
        "scale_1_exact": [name] + syns,
        "scale_2_descriptive": scale2,
        "scale_3_mechanism": scale3,
        "scale_4_domain": scale4,
        "meta": {
            "n_synonyms": len(syns),
            "n_ancestors": len(anc_ids),
        },
        "raw": raw_block,
        "err": err_block,
    }


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def _tqdm(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def main():
    ap = argparse.ArgumentParser("FULL: generate stage1 query candidates for ALL HPO terms.")
    ap.add_argument("--hpo-json", type=str, default="./data/hpo_enriched_with_llm.json")
    ap.add_argument("--out", type=str, default="./Candidate/queries.jsonl")

    # Optional slicing / sampling controls
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N terms after filtering.")
    ap.add_argument("--start", type=int, default=0, help="Start index after filtering.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle terms before slicing (use with --seed).")
    ap.add_argument("--seed", type=int, default=13)

    # LLM controls
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--stage-max-retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0)

    # Prompt files
    ap.add_argument("--system-prompt", type=str, default="./prompts/System_Prompt.txt")
    ap.add_argument("--prompt-desc", type=str, default="./prompts/stage1/Descriptive_candidates.txt")
    ap.add_argument("--prompt-mech", type=str, default="./prompts/stage1/Mechanism.txt")
    ap.add_argument("--prompt-domain", type=str, default="./prompts/stage1/Domain.tzt")

    # API config
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--api-key", type=str, default=None, help="Or use env DEEPSEEK_API_KEY")

    # Concurrency
    ap.add_argument("--workers", type=int, default=12, help="ThreadPoolExecutor workers.")
    ap.add_argument("--write-every", type=int, default=200, help="Flush to disk every N results.")

    args = ap.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing API key: set DEEPSEEK_API_KEY or pass --api-key")

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---- Step 1: Load HPO JSON
    print("[STEP 1/3] loading hpo json...")
    with open(args.hpo_json, "r", encoding="utf-8") as f:
        hpo_data = json.load(f)

    # ---- Step 2: Build indices and term list
    print("[STEP 2/3] building id->name and filtering terms...")
    id2name = build_id_to_name(hpo_data)

    if isinstance(hpo_data, dict):
        terms = [t for t in hpo_data.values() if isinstance(t, dict)]
    elif isinstance(hpo_data, list):
        terms = [t for t in hpo_data if isinstance(t, dict)]
    else:
        raise SystemExit("Unsupported HPO JSON structure.")

    # Filter terms with valid ID and name
    terms = [t for t in terms if _extract_id(t) and _extract_name(t)]
    total_terms = len(terms)

    # Optional shuffle/slice
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(terms)

    if args.start < 0:
        args.start = 0
    if args.start >= len(terms):
        raise SystemExit(f"start={args.start} exceeds number of filtered terms={len(terms)}")

    terms = terms[args.start:]
    if args.limit and args.limit > 0:
        terms = terms[:args.limit]

    print(f"[INFO] filtered terms: {total_terms} -> processing: {len(terms)} (start={args.start}, limit={args.limit})")

    # ---- Step 3: Load prompts
    print("[STEP 3/3] loading prompts...")
    system_prompt = load_text(args.system_prompt).strip()
    prompt_desc = load_text(args.prompt_desc)
    prompt_mech = load_text(args.prompt_mech)
    prompt_domain = load_text(args.prompt_domain)

    # ---- Generate with multi-thread + progress
    print("[RUN] generating candidates (multi-thread)...")
    t0 = time.time()

    # Open output file once, append incrementally (safer for long runs)
    # Write a temp file then rename at end if you prefer atomic behavior.
    out_path = args.out
    tmp_path = out_path + ".tmp"

    # If tmp exists from prior run, overwrite (test-friendly)
    with open(tmp_path, "w", encoding="utf-8") as fout:
        pass

    ok_count = 0
    fail_any_stage = 0

    # Submit jobs
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for term in terms:
            futures.append(
                ex.submit(
                    generate_for_term_safe,
                    term,
                    id2name,
                    api_key,
                    args.base_url,
                    args.model,
                    args.timeout,
                    system_prompt,
                    prompt_desc,
                    prompt_mech,
                    prompt_domain,
                    args.temperature,
                    args.stage_max_retries,
                )
            )

        # Consume results as they complete, write incrementally
        pbar = _tqdm(as_completed(futures), total=len(futures), desc="HPO terms", unit="term")

        buffer: List[str] = []
        with open(tmp_path, "a", encoding="utf-8") as fout:
            for fut in pbar:
                res = fut.result()  # safe: generate_for_term_safe never raises

                # Quick status counters
                any_err = any(bool(res.get("err", {}).get(k, "")) for k in ("scale2_err", "scale3_err", "scale4_err"))
                if any_err:
                    fail_any_stage += 1

                # Consider "ok" if it has an id+name (even if some stages empty)
                if res.get("hpo_id") and res.get("name"):
                    ok_count += 1

                # Buffer and flush
                buffer.append(json.dumps(res, ensure_ascii=False) + "\n")
                if len(buffer) >= max(1, args.write_every):
                    fout.writelines(buffer)
                    fout.flush()
                    buffer = []

                # Optional: update postfix with live stats
                if tqdm is not None:
                    pbar.set_postfix_str(f"ok={ok_count} partial_fail={fail_any_stage}")

            # final flush
            if buffer:
                fout.writelines(buffer)
                fout.flush()

    # Rename tmp -> out
    os.replace(tmp_path, out_path)

    dt = time.time() - t0
    print(f"[DONE] wrote {ok_count} items to: {out_path}")
    print(f"[STATS] partial_fail(any stage error)={fail_any_stage} / {len(terms)}")
    print(f"[TIME] {dt:.1f}s total, {dt / max(1, len(terms)):.2f}s per term")

    # Preview first line
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            print("\n[PREVIEW] first item:")
            print(json.dumps(json.loads(first), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[WARN] could not preview first item: {e}")


if __name__ == "__main__":
    main()
