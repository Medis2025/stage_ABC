#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
refill_queries.py

Goal
- Read an existing Stage-1 output JSONL: ./Candidate/queries.jsonl
- For each item, check whether any scale list is empty:
    scale_2_descriptive, scale_3_mechanism, scale_4_domain
- Count how many items are incomplete (any empty scale list).
- Refill ONLY missing scales by calling the LLM again (multi-thread + retry).
- Write a NEW jsonl with refilled results:
    ./Candidate/queries_refilled.jsonl  (default)

Notes
- This script assumes you still have the original HPO JSON available to rebuild
  definition/synonyms/ancestors blocks:
    ./data/hpo_enriched_with_llm.json
- It will NOT modify the original file; it writes a new one.
- It reuses your existing prompts:
    ./prompts/System_Prompt.txt
    ./prompts/stage1/Descriptive_candidates.txt
    ./prompts/stage1/Mechanism.txt
    ./prompts/stage1/Domain.tzt

Behavior
- If an item is already complete, it is passed through unchanged.
- If a stage refilling ultimately fails after retries, it stays empty and
  error is recorded under err fields.

Output stats
- total items
- incomplete items count
- per-scale empty counts
- after refill, remaining incomplete count
"""

import os
import sys
import json
import time
import argparse
import threading
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

# progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APIS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "APIs"))
if APIS_DIR not in sys.path:
    sys.path.insert(0, APIS_DIR)

from llm_client import LLMClient  # noqa: E402


# -----------------------------------------------------------------------------
# HPO helpers (same as your generator)
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
# Prompt helpers
# -----------------------------------------------------------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fill_template(template: str, **kwargs) -> str:
    class SafeDict(dict):
        def __missing__(self, key):
            return ""
    return template.format_map(SafeDict(**kwargs))

def parse_lines(text: str, max_lines: int = 12) -> List[str]:
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
# Retry wrapper (treat empty output as error -> triggers retry)
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
    empty_is_error: bool = True,
) -> Tuple[str, Optional[str], bool]:
    """
    Returns: (raw_text, err_str, had_retry)
    """
    last_err: Optional[str] = None
    had_retry = False

    for attempt in range(max_retries + 1):
        try:
            raw = llm.run(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)

            if empty_is_error and (raw is None or str(raw).strip() == ""):
                raise RuntimeError("empty llm output")

            if attempt > 0:
                had_retry = True
            return str(raw), None, had_retry

        except Exception as e:
            if attempt > 0:
                had_retry = True

            last_err = f"{type(e).__name__}: {e}"

            if attempt >= max_retries:
                return "", last_err, had_retry

            sleep_s = min(max_sleep, base_sleep * (2 ** attempt))
            sleep_s = sleep_s + (0.1 * ((attempt % 3) + 1))
            time.sleep(sleep_s)

    return "", last_err, had_retry


# -----------------------------------------------------------------------------
# Thread-local LLM client
# -----------------------------------------------------------------------------

_THREAD_LOCAL = threading.local()

def get_thread_llm(api_key: str, base_url: str, model: str, timeout: float) -> LLMClient:
    llm = getattr(_THREAD_LOCAL, "llm", None)
    if llm is None:
        llm = LLMClient(api_key=api_key, base_url=base_url, model=model, timeout=timeout)
        _THREAD_LOCAL.llm = llm
    return llm


# -----------------------------------------------------------------------------
# JSONL IO
# -----------------------------------------------------------------------------

def read_jsonl_list(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception:
                # skip broken line
                continue
    return items

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Completeness checks + blocks for refill
# -----------------------------------------------------------------------------

def _is_empty_list(x: Any) -> bool:
    return (x is None) or (not isinstance(x, list)) or (len(x) == 0)

def item_missing_scales(item: Dict[str, Any]) -> Dict[str, bool]:
    """
    Return flags for missing scale2/3/4.
    """
    return {
        "scale2": _is_empty_list(item.get("scale_2_descriptive")),
        "scale3": _is_empty_list(item.get("scale_3_mechanism")),
        "scale4": _is_empty_list(item.get("scale_4_domain")),
    }

def is_incomplete(item: Dict[str, Any]) -> bool:
    m = item_missing_scales(item)
    return m["scale2"] or m["scale3"] or m["scale4"]

def build_term_maps(hpo_json_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Returns:
      term_by_id: hpo_id -> original term dict
      id2name: hpo_id -> name
    """
    with open(hpo_json_path, "r", encoding="utf-8") as f:
        hpo_data = json.load(f)

    id2name = build_id_to_name(hpo_data)

    if isinstance(hpo_data, dict):
        terms = [t for t in hpo_data.values() if isinstance(t, dict)]
    elif isinstance(hpo_data, list):
        terms = [t for t in hpo_data if isinstance(t, dict)]
    else:
        raise SystemExit("Unsupported HPO JSON structure.")

    terms = [t for t in terms if _extract_id(t) and _extract_name(t)]
    term_by_id = { _extract_id(t): t for t in terms }
    return term_by_id, id2name


def build_blocks_for_hid(hid: str, term_by_id: Dict[str, Dict[str, Any]], id2name: Dict[str, str]) -> Optional[Dict[str, Any]]:
    term = term_by_id.get(hid)
    if not term:
        return None

    name = _extract_name(term)
    syns = _extract_synonyms(term)
    definition = _extract_def(term)
    llm_add_def = _extract_llm_add_def(term)
    anc_ids = _extract_ancestor_ids(term)
    anc_names = [id2name.get(x, x) for x in anc_ids]

    return {
        "hpo_id": hid,
        "name": name,
        "syns": syns,
        "synonyms_block": "\n".join(syns) if syns else "",
        "def_block": definition if definition else "",
        "llm_add_def_block": llm_add_def if llm_add_def else "",
        "ancestor_names_block": "\n".join(anc_names) if anc_names else "",
        "n_ancestors": len(anc_ids),
    }


# -----------------------------------------------------------------------------
# Refill worker: refill only missing stages for one item
# -----------------------------------------------------------------------------

def refill_one_item(
    item: Dict[str, Any],
    blocks: Dict[str, Any],
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
    If scale2/3/4 is empty, refill it. Otherwise keep it.
    Never raises.
    """
    hid = blocks["hpo_id"]
    name = blocks["name"]

    # ensure keys exist
    item.setdefault("hpo_id", hid)
    item.setdefault("name", name)
    item.setdefault("raw", {})
    item.setdefault("err", {})
    item.setdefault("retry", {})
    item.setdefault("meta", {})
    item.setdefault("scale_2_descriptive", [])
    item.setdefault("scale_3_mechanism", [])
    item.setdefault("scale_4_domain", [])

    llm = get_thread_llm(api_key=api_key, base_url=base_url, model=model, timeout=timeout)

    missing = item_missing_scales(item)

    # scale2
    if missing["scale2"]:
        user_desc = fill_template(
            prompt_desc,
            hpo_id=hid,
            name=name,
            synonyms_block=blocks["synonyms_block"],
            def_block=blocks["def_block"],
        )
        raw_desc, err_desc, retry_desc = llm_run_with_retry(
            llm, system_prompt, user_desc, temperature, max_retries=stage_max_retries
        )
        if raw_desc:
            item["scale_2_descriptive"] = parse_lines(raw_desc)
        item["raw"]["scale2"] = raw_desc
        item["err"]["scale2_err"] = err_desc or ""
        item["retry"]["scale2_retry"] = bool(retry_desc)

    # scale3
    if missing["scale3"]:
        user_mech = fill_template(
            prompt_mech,
            hpo_id=hid,
            name=name,
            def_block=blocks["def_block"],
            llm_add_def_block=blocks["llm_add_def_block"],
        )
        raw_mech, err_mech, retry_mech = llm_run_with_retry(
            llm, system_prompt, user_mech, temperature, max_retries=stage_max_retries
        )
        if raw_mech:
            item["scale_3_mechanism"] = parse_lines(raw_mech)
        item["raw"]["scale3"] = raw_mech
        item["err"]["scale3_err"] = err_mech or ""
        item["retry"]["scale3_retry"] = bool(retry_mech)

    # scale4
    if missing["scale4"]:
        user_dom = fill_template(
            prompt_domain,
            hpo_id=hid,
            name=name,
            ancestor_names_block=blocks["ancestor_names_block"],
        )
        raw_dom, err_dom, retry_dom = llm_run_with_retry(
            llm, system_prompt, user_dom, temperature, max_retries=stage_max_retries
        )
        if raw_dom:
            item["scale_4_domain"] = parse_lines(raw_dom)
        item["raw"]["scale4"] = raw_dom
        item["err"]["scale4_err"] = err_dom or ""
        item["retry"]["scale4_retry"] = bool(retry_dom)

    # refresh meta if missing
    item["meta"]["n_synonyms"] = item["meta"].get("n_synonyms", len(blocks["syns"]))
    item["meta"]["n_ancestors"] = item["meta"].get("n_ancestors", blocks.get("n_ancestors", 0))

    # ensure scale_1_exact exists
    if "scale_1_exact" not in item or not isinstance(item["scale_1_exact"], list) or len(item["scale_1_exact"]) == 0:
        item["scale_1_exact"] = [name] + blocks["syns"]

    return item


# -----------------------------------------------------------------------------
# Progress wrapper
# -----------------------------------------------------------------------------

def _tqdm(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Refill empty scale fields from an existing queries.jsonl and save to new jsonl.")
    ap.add_argument("--in-jsonl", type=str, default="./Candidate/queries.jsonl")
    ap.add_argument("--out-jsonl", type=str, default="./Candidate/queries_refilled.jsonl")

    ap.add_argument("--hpo-json", type=str, default="./data/hpo_enriched_with_llm.json")

    # prompts
    ap.add_argument("--system-prompt", type=str, default="./prompts/System_Prompt.txt")
    ap.add_argument("--prompt-desc", type=str, default="./prompts/stage1/Descriptive_candidates.txt")
    ap.add_argument("--prompt-mech", type=str, default="./prompts/stage1/Mechanism.txt")
    ap.add_argument("--prompt-domain", type=str, default="./prompts/stage1/Domain.tzt")

    # llm/api
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--api-key", type=str, default=None, help="Or use env DEEPSEEK_API_KEY")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--stage-max-retries", type=int, default=5)

    # concurrency
    ap.add_argument("--workers", type=int, default=12)

    args = ap.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing API key: set DEEPSEEK_API_KEY or pass --api-key")

    if not os.path.exists(args.in_jsonl):
        raise SystemExit(f"❌ input jsonl not found: {args.in_jsonl}")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    print("[STEP 1/4] load existing jsonl...")
    items = read_jsonl_list(args.in_jsonl)
    if not items:
        raise SystemExit("❌ input jsonl is empty or unreadable")

    # stats before refill
    total = len(items)
    missing_flags = [item_missing_scales(it) for it in items]
    missing_any = sum(1 for m in missing_flags if (m["scale2"] or m["scale3"] or m["scale4"]))
    missing_s2 = sum(1 for m in missing_flags if m["scale2"])
    missing_s3 = sum(1 for m in missing_flags if m["scale3"])
    missing_s4 = sum(1 for m in missing_flags if m["scale4"])

    print("[STATS before]")
    print(f"  total items: {total}")
    print(f"  incomplete items (any empty scale): {missing_any}")
    print(f"    empty scale2: {missing_s2}")
    print(f"    empty scale3: {missing_s3}")
    print(f"    empty scale4: {missing_s4}")

    print("[STEP 2/4] load hpo json and build term maps...")
    term_by_id, id2name = build_term_maps(args.hpo_json)

    print("[STEP 3/4] load prompts...")
    system_prompt = load_text(args.system_prompt).strip()
    prompt_desc = load_text(args.prompt_desc)
    prompt_mech = load_text(args.prompt_mech)
    prompt_domain = load_text(args.prompt_domain)

    # Build tasks for incomplete items
    # Keep original order; we'll fill in-place and then write new file.
    tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, it in enumerate(items):
        hid = it.get("hpo_id") or ""
        if not hid:
            continue
        if not is_incomplete(it):
            continue
        blocks = build_blocks_for_hid(hid, term_by_id, id2name)
        if blocks is None:
            continue
        tasks.append((idx, it, blocks))

    print(f"[STEP 4/4] refill incomplete items: {len(tasks)} (workers={args.workers})")

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = []
            for idx, it, blk in tasks:
                futures.append(
                    ex.submit(
                        refill_one_item,
                        it,
                        blk,
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

            pbar = _tqdm(as_completed(futures), total=len(futures), desc="refill", unit="item")
            # Consume futures; items are mutated in-place by reference, but we also assign back.
            for fut in pbar:
                updated = fut.result()
                # no-op: already updated in-place; keep for clarity
                _ = updated

    # stats after refill
    missing_flags2 = [item_missing_scales(it) for it in items]
    missing_any2 = sum(1 for m in missing_flags2 if (m["scale2"] or m["scale3"] or m["scale4"]))
    missing_s22 = sum(1 for m in missing_flags2 if m["scale2"])
    missing_s32 = sum(1 for m in missing_flags2 if m["scale3"])
    missing_s42 = sum(1 for m in missing_flags2 if m["scale4"])

    print("[STATS after]")
    print(f"  total items: {total}")
    print(f"  incomplete items (any empty scale): {missing_any2}")
    print(f"    empty scale2: {missing_s22}")
    print(f"    empty scale3: {missing_s32}")
    print(f"    empty scale4: {missing_s42}")

    print(f"[WRITE] saving to: {args.out_jsonl}")
    write_jsonl(args.out_jsonl, items)

    # preview
    try:
        # show one incomplete if any, else show first
        preview = None
        for it in items:
            if is_incomplete(it):
                preview = it
                break
        if preview is None and items:
            preview = items[0]
        if preview is not None:
            print("\n[PREVIEW] one item:")
            print(json.dumps(preview, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[WARN] preview failed: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
