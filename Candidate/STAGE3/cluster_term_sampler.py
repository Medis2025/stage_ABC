#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_term_sampler.py  (REVISED)

Goal:
- Build PubMed query terms for validating Stage2 clusters (co-occurrence plausibility)
- Optional: check ESearch hits
- Optional: LLM repair (DeepSeek) ONLY when triggered and LOW-FREQUENCY

Revisions (per your request)
1) Normalize casing: ALL phrases normalized to lower-case for reproducibility audit.
2) Negative baseline repair constraint:
   - Only allow repairing the SEED phrase (standardization).
   - Do NOT repair the random non-neighbor phrase (avoid "strengthening" the negative).
3) Persist hit-count audit fields in output jsonl:
   - query + hits
   - repaired_query + repaired_hits
   - used_query + used_hits

LLM role:
- Repair only (optional, low frequency)
- Trigger conditions:
  - phrase judged "weak" OR
  - ESearch hits == 0 (when --check_esearch enabled)

Output:
- console prints grouped queries
- optional --write_terms_jsonl writes structured records for PhaseA batch driver

Typical run:
python cluster_term_sampler.py \
  --queries_jsonl "$QUERIES_JSONL" \
  --neighbors_jsonl "$NEIGHBORS_JSONL" \
  --seed_hpo HP:0000009 \
  --seed 42 \
  --n_seed1 3 --n_seed2 3 --n_neg 3 \
  --check_esearch \
  --email "xxx@qq.com" \
  --use_llm_repair \
  --llm_model deepseek-chat \
  --llm_base_url https://api.deepseek.com \
  --write_terms_jsonl ./terms_esearch_llm.jsonl
"""

from __future__ import annotations

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# Use your local NCBI helper
from pubmed_pmc_client import PubMedPMCClient, NCBIConfig  # type: ignore

# Optional DeepSeek client (OpenAI-compatible) provided by you
try:
    from llm_client import LLMClient  # type: ignore
    _HAVE_LLM = True
except Exception:
    LLMClient = None
    _HAVE_LLM = False


# -----------------------------
# IO
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Text normalize
# -----------------------------

def _clean_phrase(s: str) -> str:
    s = " ".join((s or "").strip().split())
    return s

def _norm_phrase(s: str) -> str:
    """
    Reproducibility: lower-case, compact whitespace.
    """
    return _clean_phrase(s).lower()

def _quote(s: str) -> str:
    return f"\"{s}\""


# -----------------------------
# Weak phrase heuristics (cheap + deterministic)
# -----------------------------

_WEAK_SUBSTRINGS = {
    "signs", "symptoms", "with", "without", "lack of", "near", "due to", "induced",
    "presenting", "diagnosed", "patient", "patients", "case", "clinical",
    "pathophysiology", "mechanism", "associated with", "in the setting of"
}

def is_weak_phrase(phrase: str, max_tokens: int = 5) -> bool:
    """
    Deterministic weak-phrase detection.
    Trigger repair when:
      - too long
      - contains narrative / non-term patterns
      - contains parentheses or too many commas
    """
    p = _norm_phrase(phrase)
    if not p:
        return True
    toks = p.split()
    if len(toks) > max_tokens:
        return True
    if "(" in p or ")" in p:
        return True
    if p.count(",") >= 1:
        return True
    for w in _WEAK_SUBSTRINGS:
        if w in p:
            return True
    return False


# -----------------------------
# PubMed ESearch wrapper
# -----------------------------

def pubmed_esearch_hits(client: PubMedPMCClient, term: str, retmax: int = 0) -> int:
    """
    Returns hit count (best-effort).
    PubMedPMCClient.pubmed_esearch(term, retmax=...) is assumed to exist.
    """
    try:
        r = client.pubmed_esearch(term, retmax=retmax)
        if not r or not r.get("ok"):
            return 0
        # Some clients return "count", some don't; fallback:
        if "count" in r and isinstance(r["count"], int):
            return int(r["count"])
        pmids = r.get("pmids") or []
        # If count not provided, approximate by pmids length (lower bound)
        return int(len(pmids))
    except Exception:
        return 0


# -----------------------------
# LLM repair (plain text only)
# -----------------------------

def _repair_prompt(phrase: str) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    Output MUST be plain text (no JSON), single phrase, lower-case.
    """
    system_prompt = (
        "You are a biomedical term normalizer. "
        "You rewrite a weak clinical phrase into a short standard biomedical keyword phrase."
    )
    user_prompt = (
        "Rewrite the phrase into ONE short standard biomedical keyword phrase.\n"
        "Rules:\n"
        "- output plain text only\n"
        "- one line only\n"
        "- lower-case only\n"
        "- 2 to 5 words\n"
        "- no quotes, no punctuation\n"
        "- keep the meaning; prefer common PubMed terms\n\n"
        f"phrase: {phrase}"
    )
    return system_prompt, user_prompt

def llm_repair_phrase(llm: LLMClient, phrase: str) -> str:
    phrase_in = _norm_phrase(phrase)
    if not phrase_in:
        return phrase_in
    sys_p, user_p = _repair_prompt(phrase_in)
    out = (llm.run(sys_p, user_p, temperature=0.2) or "").strip()
    out = _norm_phrase(out)
    # Safety guard: if model returns multiple lines, take first non-empty
    if "\n" in out:
        out = _norm_phrase(out.splitlines()[0])
    # If it returns empty, fallback
    return out or phrase_in


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TermRecord:
    kind: str                   # seed0 / seed1 / seed2 / neg
    seed_hpo: str
    neighbor_hpos: List[str]
    phrases: Dict[str, str]     # seed / n1 / n2 / random
    query: str
    hits: Optional[int] = None
    repaired: bool = False
    repaired_phrases: Optional[Dict[str, str]] = None
    repaired_query: Optional[str] = None
    repaired_hits: Optional[int] = None
    used_query: Optional[str] = None
    used_hits: Optional[int] = None


# -----------------------------
# Core Sampler
# -----------------------------

class ClusterTermSampler:
    """
    Build PubMed terms from:
      - Layer0 queries_refilled.jsonl
      - Layer2 neighbors.jsonl topology

    Repair policy:
      - ONLY Repair (optional, low frequency)
      - Triggered by weak-phrase OR 0 hits (if --check_esearch)
      - Negative baseline: repair seed only, DO NOT repair random phrase
    """

    def __init__(
        self,
        queries_jsonl: str,
        neighbors_jsonl: str,
        *,
        seed: int = 13,
        use_case_filter: bool = True,
        case_filter: str = "(patient OR patients OR case OR clinical OR presenting OR diagnosed)",
        allow_scales: Tuple[str, ...] = ("scale3", "scale2", "scale1"),
        neighbor_cosine_floor: float = 0.001,
    ):
        self.rng = random.Random(seed)
        self.use_case_filter = use_case_filter
        self.case_filter = case_filter
        self.allow_scales = allow_scales
        self.neighbor_cosine_floor = float(neighbor_cosine_floor)

        self.hpo2phrases = self._load_queries(queries_jsonl)
        self.seed2neighbors = self._load_neighbors(neighbors_jsonl)

        self._all_hpos = list(self.hpo2phrases.keys())

    # -------------------------
    # Loading
    # -------------------------

    def _load_queries(self, path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        hpo_id -> {scale1, scale2, scale3}
        """
        data: Dict[str, Dict[str, List[str]]] = {}
        for obj in read_jsonl(path):
            hid = obj.get("hpo_id")
            if not hid:
                continue
            hid = str(hid).strip()
            data[hid] = {
                "scale1": [_norm_phrase(x) for x in (obj.get("scale_1_exact") or []) if _clean_phrase(x)],
                "scale2": [_norm_phrase(x) for x in (obj.get("scale_2_descriptive") or []) if _clean_phrase(x)],
                "scale3": [_norm_phrase(x) for x in (obj.get("scale_3_mechanism") or []) if _clean_phrase(x)],
            }
        return data

    def _load_neighbors(self, path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        seed_hpo -> [{hpo_id, cosine, rank}]
        """
        out: Dict[str, List[Dict[str, Any]]] = {}
        for obj in read_jsonl(path):
            seed = str(obj["seed_hpo"]).strip()
            out[seed] = obj.get("neighbors") or []
        return out

    # -------------------------
    # Sampling primitives
    # -------------------------

    def _pick_phrase(self, hid: str) -> Optional[str]:
        ph = self.hpo2phrases.get(hid)
        if not ph:
            return None
        for key in self.allow_scales:
            lst = ph.get(key) or []
            if lst:
                # already normalized lower-case
                return self.rng.choice(lst)
        return None

    def _sample_neighbor_hpo(self, seed_hpo: str) -> Optional[str]:
        neighs = self.seed2neighbors.get(seed_hpo) or []
        if not neighs:
            return None
        weights = [max(self.neighbor_cosine_floor, float(n.get("cosine", 0.0))) for n in neighs]
        return str(self.rng.choices(neighs, weights=weights, k=1)[0]["hpo_id"]).strip()

    def _sample_two_neighbors(self, seed_hpo: str) -> Optional[Tuple[str, str]]:
        neighs = self.seed2neighbors.get(seed_hpo) or []
        if len(neighs) < 2:
            return None
        weights = [max(self.neighbor_cosine_floor, float(n.get("cosine", 0.0))) for n in neighs]
        picks = self.rng.choices(neighs, weights=weights, k=2)
        n1 = str(picks[0]["hpo_id"]).strip()
        n2 = str(picks[1]["hpo_id"]).strip()
        return n1, n2

    def _sample_random_non_neighbor(self, seed_hpo: str) -> Optional[str]:
        """
        Random HPO that is not in top-K neighbors of seed, and not itself.
        """
        neighs = self.seed2neighbors.get(seed_hpo) or []
        forbid = {seed_hpo}
        for n in neighs:
            forbid.add(str(n.get("hpo_id", "")).strip())
        candidates = [h for h in self._all_hpos if h not in forbid]
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _append_case_filter(self, q: str) -> str:
        if not self.use_case_filter:
            return q
        return f"{q} AND {self.case_filter}"

    # -------------------------
    # Query builders (normalized lower-case phrases)
    # -------------------------

    def build_seed0(self, seed_hpo: str) -> Optional[TermRecord]:
        p0 = self._pick_phrase(seed_hpo)
        if not p0:
            return None
        q = self._append_case_filter(_quote(p0))
        return TermRecord(
            kind="seed0",
            seed_hpo=seed_hpo,
            neighbor_hpos=[],
            phrases={"seed": p0},
            query=q,
        )

    def build_seed1(self, seed_hpo: str) -> Optional[TermRecord]:
        nb = self._sample_neighbor_hpo(seed_hpo)
        if not nb:
            return None
        p0 = self._pick_phrase(seed_hpo)
        p1 = self._pick_phrase(nb)
        if not p0 or not p1:
            return None
        q = self._append_case_filter(f"{_quote(p0)} AND {_quote(p1)}")
        return TermRecord(
            kind="seed1",
            seed_hpo=seed_hpo,
            neighbor_hpos=[nb],
            phrases={"seed": p0, "n1": p1},
            query=q,
        )

    def build_seed2(self, seed_hpo: str) -> Optional[TermRecord]:
        pair = self._sample_two_neighbors(seed_hpo)
        if not pair:
            return None
        n1, n2 = pair
        p0 = self._pick_phrase(seed_hpo)
        p1 = self._pick_phrase(n1)
        p2 = self._pick_phrase(n2)
        if not p0 or not p1 or not p2:
            return None
        q = self._append_case_filter(f"{_quote(p0)} AND ({_quote(p1)} OR {_quote(p2)})")
        return TermRecord(
            kind="seed2",
            seed_hpo=seed_hpo,
            neighbor_hpos=[n1, n2],
            phrases={"seed": p0, "n1": p1, "n2": p2},
            query=q,
        )

    def build_neg(self, seed_hpo: str) -> Optional[TermRecord]:
        rnd = self._sample_random_non_neighbor(seed_hpo)
        if not rnd:
            return None
        p0 = self._pick_phrase(seed_hpo)
        pr = self._pick_phrase(rnd)
        if not p0 or not pr:
            return None
        q = self._append_case_filter(f"{_quote(p0)} AND {_quote(pr)}")
        return TermRecord(
            kind="neg",
            seed_hpo=seed_hpo,
            neighbor_hpos=[rnd],
            phrases={"seed": p0, "random": pr},
            query=q,
        )

    # -------------------------
    # Repair logic
    # -------------------------

    def maybe_repair(
        self,
        rec: TermRecord,
        *,
        llm: Optional[LLMClient],
        client: Optional[PubMedPMCClient],
        check_esearch: bool,
        retmax_for_check: int = 0,
        allow_repair: bool = True,
    ) -> TermRecord:
        """
        Mutates rec with hits, repaired_* fields, and chooses used_query.
        """
        # 1) compute hits for original (if requested)
        if check_esearch and client is not None:
            rec.hits = pubmed_esearch_hits(client, rec.query, retmax=retmax_for_check)

        # decide if repair needed
        need_repair = False
        if allow_repair and llm is not None:
            # weak phrase trigger
            for k, ph in rec.phrases.items():
                # negative baseline: do NOT consider random phrase for repair trigger
                if rec.kind == "neg" and k == "random":
                    continue
                if is_weak_phrase(ph):
                    need_repair = True
                    break
            # 0-hit trigger
            if check_esearch and (rec.hits is not None) and rec.hits == 0:
                need_repair = True

        if not need_repair or llm is None:
            # choose used as original
            rec.used_query = rec.query
            rec.used_hits = rec.hits
            return rec

        # 2) perform repair on allowed parts only
        repaired_ph: Dict[str, str] = dict(rec.phrases)

        # policy:
        # - seed0/seed1/seed2: can repair seed and neighbors
        # - neg: repair seed ONLY; keep random phrase unchanged
        for k, ph in rec.phrases.items():
            if rec.kind == "neg" and k == "random":
                continue
            # repair only if weak OR (0-hit and this is seed or neighbor)
            if is_weak_phrase(ph) or (check_esearch and (rec.hits == 0 if rec.hits is not None else False)):
                repaired_ph[k] = llm_repair_phrase(llm, ph)

        # 3) rebuild repaired query string
        rq = None
        if rec.kind == "seed0":
            rq = self._append_case_filter(_quote(repaired_ph["seed"]))
        elif rec.kind == "seed1":
            rq = self._append_case_filter(f"{_quote(repaired_ph['seed'])} AND {_quote(repaired_ph['n1'])}")
        elif rec.kind == "seed2":
            rq = self._append_case_filter(
                f"{_quote(repaired_ph['seed'])} AND ({_quote(repaired_ph['n1'])} OR {_quote(repaired_ph['n2'])})"
            )
        elif rec.kind == "neg":
            # IMPORTANT: random part unchanged by policy
            rq = self._append_case_filter(f"{_quote(repaired_ph['seed'])} AND {_quote(repaired_ph['random'])}")

        rec.repaired = True
        rec.repaired_phrases = repaired_ph
        rec.repaired_query = rq

        # 4) compute repaired hits (if requested)
        if check_esearch and client is not None and rq:
            rec.repaired_hits = pubmed_esearch_hits(client, rq, retmax=retmax_for_check)

        # 5) choose used_query
        # rule: if repaired_hits > 0 and original_hits == 0 -> use repaired
        # else if repaired_hits >= original_hits -> use repaired (mildly optimistic)
        # else use original
        rec.used_query = rec.query
        rec.used_hits = rec.hits

        if check_esearch and client is not None and rq:
            oh = rec.hits if rec.hits is not None else 0
            rh = rec.repaired_hits if rec.repaired_hits is not None else 0
            if (oh == 0 and rh > 0) or (rh >= oh and rh > 0):
                rec.used_query = rq
                rec.used_hits = rh
        else:
            # without esearch, if repair happened, prefer repaired
            rec.used_query = rq or rec.query
            rec.used_hits = rec.repaired_hits if rec.repaired_hits is not None else rec.hits

        return rec


# -----------------------------
# CLI main
# -----------------------------

def build_llm_from_args(args) -> Optional[LLMClient]:
    if not args.use_llm_repair:
        return None
    if not _HAVE_LLM:
        raise RuntimeError("llm_client.py not importable, but --use_llm_repair was set.")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment.")
    return LLMClient(
        api_key=api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout=float(args.llm_timeout),
    )

def build_pubmed_client_from_args(args) -> Optional[PubMedPMCClient]:
    if not args.check_esearch:
        return None
    cfg = NCBIConfig(
        email=args.email,
        tool=args.tool,
        api_key=(args.api_key or None),
        polite_sleep=float(args.polite_sleep),
    )
    return PubMedPMCClient(cfg)

def rec_to_json(rec: TermRecord) -> Dict[str, Any]:
    return {
        "kind": rec.kind,
        "seed_hpo": rec.seed_hpo,
        "neighbor_hpos": rec.neighbor_hpos,
        "phrases": rec.phrases,
        "query": rec.query,
        "hits": rec.hits,
        "repaired": rec.repaired,
        "repaired_phrases": rec.repaired_phrases,
        "repaired_query": rec.repaired_query,
        "repaired_hits": rec.repaired_hits,
        "used_query": rec.used_query,
        "used_hits": rec.used_hits,
    }

def main():
    ap = argparse.ArgumentParser("cluster_term_sampler")
    ap.add_argument("--queries_jsonl", type=str, required=True)
    ap.add_argument("--neighbors_jsonl", type=str, required=True)

    ap.add_argument("--seed_hpo", type=str, required=True)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--n_seed0", type=int, default=1)
    ap.add_argument("--n_seed1", type=int, default=3)
    ap.add_argument("--n_seed2", type=int, default=3)
    ap.add_argument("--n_neg", type=int, default=3)

    ap.add_argument("--use_case_filter", action="store_true", default=True)
    ap.add_argument("--case_filter", type=str, default="(patient OR patients OR case OR clinical OR presenting OR diagnosed)")

    # ESearch check
    ap.add_argument("--check_esearch", action="store_true")
    ap.add_argument("--email", type=str, default="")
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--tool", type=str, default="hpo-agent-stage3-sampler")
    ap.add_argument("--polite_sleep", type=float, default=0.34)

    # LLM repair
    ap.add_argument("--use_llm_repair", action="store_true")
    ap.add_argument("--llm_model", type=str, default="deepseek-chat")
    ap.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--llm_timeout", type=float, default=60.0)

    # Output
    ap.add_argument("--write_terms_jsonl", type=str, default="")

    args = ap.parse_args()

    sampler = ClusterTermSampler(
        queries_jsonl=args.queries_jsonl,
        neighbors_jsonl=args.neighbors_jsonl,
        seed=int(args.seed),
        use_case_filter=bool(args.use_case_filter),
        case_filter=str(args.case_filter),
    )

    llm = build_llm_from_args(args) if args.use_llm_repair else None
    pubmed_client = build_pubmed_client_from_args(args) if args.check_esearch else None

    all_recs: List[TermRecord] = []

    seed_hpo = args.seed_hpo.strip()

    # Seed+0
    print("\n[Seed + 0]")
    for _ in range(max(0, int(args.n_seed0))):
        r = sampler.build_seed0(seed_hpo)
        if not r:
            continue
        r = sampler.maybe_repair(r, llm=llm, client=pubmed_client, check_esearch=bool(args.check_esearch))
        print(r.used_query or r.query)
        if r.repaired_query and (r.used_query == r.repaired_query):
            print(f"  -> repaired: {r.repaired_query}")
        all_recs.append(r)

    # Seed+1
    print("\n[Seed + 1]")
    for _ in range(max(0, int(args.n_seed1))):
        r = sampler.build_seed1(seed_hpo)
        if not r:
            continue
        r = sampler.maybe_repair(r, llm=llm, client=pubmed_client, check_esearch=bool(args.check_esearch))
        print(r.used_query or r.query)
        if r.repaired_query and (r.used_query == r.repaired_query):
            print(f"  -> repaired: {r.repaired_query}")
        all_recs.append(r)

    # Seed+2
    print("\n[Seed + 2]")
    for _ in range(max(0, int(args.n_seed2))):
        r = sampler.build_seed2(seed_hpo)
        if not r:
            continue
        r = sampler.maybe_repair(r, llm=llm, client=pubmed_client, check_esearch=bool(args.check_esearch))
        print(r.used_query or r.query)
        if r.repaired_query and (r.used_query == r.repaired_query):
            print(f"  -> repaired: {r.repaired_query}")
        all_recs.append(r)

    # Negative baseline
    print("\n[Negative baseline: seed + random non-neighbor]")
    for _ in range(max(0, int(args.n_neg))):
        r = sampler.build_neg(seed_hpo)
        if not r:
            continue
        # IMPORTANT: negative baseline repair policy is enforced in maybe_repair()
        r = sampler.maybe_repair(r, llm=llm, client=pubmed_client, check_esearch=bool(args.check_esearch))
        print(r.used_query or r.query)
        if r.repaired_query and (r.used_query == r.repaired_query):
            print(f"  -> repaired: {r.repaired_query}")
        all_recs.append(r)

    # Write jsonl
    if args.write_terms_jsonl:
        rows = [rec_to_json(r) for r in all_recs]
        write_jsonl(args.write_terms_jsonl, rows)
        print(f"\n[WROTE] terms jsonl -> {args.write_terms_jsonl} (n={len(rows)})")

if __name__ == "__main__":
    main()
