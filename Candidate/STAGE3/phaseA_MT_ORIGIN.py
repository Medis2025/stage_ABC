#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_stage3_phaseA_full.py  (REVISED: resume/rerun-safe + cache-in-output-dir, minimized changes)

Minimized changes on top of your resume-safe version:
1) Cache defaults to output dir (run_dir/.cache) WITHOUT requiring --cache_dir
   - If --cache_dir is provided, it overrides default.
2) Ensure cache_dir exists.
Everything else unchanged.
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import threading
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Iterable, DefaultDict, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from cluster_term_sampler import ClusterTermSampler, TermRecord  # type: ignore
from pubmed_pmc_client import PubMedPMCClient, NCBIConfig, _normalize_pmcid  # type: ignore


# =============================================================================
# IO helpers
# =============================================================================

def ensure_dir(p: str) -> None:
    if not p:
        return
    os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl_iter(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                # allow broken last line after kill -9
                continue

def compact_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _atomic_write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def qkey(query: str, retmax: int) -> str:
    return _sha1(f"{query}\n{int(retmax)}")


# =============================================================================
# Global NCBI rate limiter (thread-safe)
# =============================================================================

class GlobalRateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.time()
            self._next_allowed = now + self.min_interval


_RATE_LIMITER: Optional[GlobalRateLimiter] = None

def _rate_limited_call(fn, *args, **kwargs):
    global _RATE_LIMITER
    if _RATE_LIMITER is not None:
        _RATE_LIMITER.wait()
    return fn(*args, **kwargs)


# =============================================================================
# PhaseA extractor (kept same as your file: only need "from_xml" path)
# =============================================================================

_SENT_SPLIT = re.compile(r"([.!?;。！？；](?:\s+|$)|\n+)")
_CLAUSE_SPLIT = re.compile(r"(,\s+|;\s+|:\s+)")

def iter_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not text:
        return spans
    parts = _SENT_SPLIT.split(text)
    cur = ""
    cur_start = 0
    cur_pos = 0
    tmp: List[str] = [p for p in parts if p]
    for p in tmp:
        if _SENT_SPLIT.match(p):
            cur += p
            cur_pos += len(p)
            st = cur_start
            ed = st + len(cur)
            spans.append((st, ed))
            cur = ""
            cur_start = cur_pos
        else:
            cur += p
            cur_pos += len(p)
    if cur.strip():
        st = cur_start
        ed = st + len(cur)
        spans.append((st, ed))

    merged: List[Tuple[int, int]] = []
    for st, ed in spans:
        if not merged:
            merged.append((st, ed))
            continue
        if ed - st < 25:
            pst, _ = merged[-1]
            merged[-1] = (pst, ed)
        else:
            merged.append((st, ed))
    return merged

def _pick_containing_span(spans: List[Tuple[int, int]], a: int, b: int) -> Tuple[int, int]:
    for st, ed in spans:
        if st <= a and b <= ed:
            return st, ed
    mid = (a + b) / 2.0
    return min(spans, key=lambda x: abs(((x[0] + x[1]) / 2.0) - mid))

def _clause_first_within_sentence(sentence: str, rel_span: Tuple[int, int], side_extra: int = 40) -> str:
    if not sentence:
        return ""
    a, b = rel_span
    a = max(0, a)
    b = min(len(sentence), b)

    parts = _CLAUSE_SPLIT.split(sentence)
    clauses: List[Tuple[int, int]] = []
    buf = ""
    buf_start = 0
    cur_pos = 0
    tmp: List[str] = [p for p in parts if p]

    for p in tmp:
        if _CLAUSE_SPLIT.match(p):
            buf += p
            cur_pos += len(p)
            st = buf_start
            ed = st + len(buf)
            clauses.append((st, ed))
            buf = ""
            buf_start = cur_pos
        else:
            buf += p
            cur_pos += len(p)

    if buf.strip():
        st = buf_start
        ed = st + len(buf)
        clauses.append((st, ed))

    if not clauses:
        return compact_ws(sentence)

    cst, ced = _pick_containing_span(clauses, a, b)
    st2 = max(0, cst - side_extra)
    ed2 = min(len(sentence), ced + side_extra)
    return compact_ws(sentence[st2:ed2])

def sentence_first_context(paragraph: str, match_span: Tuple[int, int],
                           side_extra: int = 60,
                           clause_first: bool = False) -> str:
    if not paragraph:
        return ""
    a, b = match_span
    a = max(0, a)
    b = min(len(paragraph), b)

    spans = iter_sentence_spans(paragraph)
    if not spans:
        st = max(0, a - 120)
        ed = min(len(paragraph), b + 120)
        return compact_ws(paragraph[st:ed])

    sst, sed = _pick_containing_span(spans, a, b)
    sent = paragraph[sst:sed]

    if clause_first:
        rel = (a - sst, b - sst)
        return _clause_first_within_sentence(sent, rel_span=rel, side_extra=min(40, side_extra))

    return compact_ws(sent)

_LOW_WORDS = re.compile(r"\b(low|decreased|reduced|hypo|below|lowered)\b", re.IGNORECASE)
_HIGH_WORDS = re.compile(r"\b(high|elevated|increased|hyper|above|raised)\b", re.IGNORECASE)
_NORMAL_WORDS = re.compile(r"\b(normal|within\s+normal|reference\s+range)\b", re.IGNORECASE)

def polarity_flags(ctx: str) -> List[str]:
    flags: List[str] = []
    if not ctx:
        return flags
    if _LOW_WORDS.search(ctx):
        flags.append("LOW_FLAG")
    if _HIGH_WORDS.search(ctx):
        flags.append("HIGH_FLAG")
    if _NORMAL_WORDS.search(ctx):
        flags.append("NORMAL_FLAG")
    return flags

_RE_BP = re.compile(r"\b(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})\s*(?P<unit>mmhg)\b", re.IGNORECASE)
_RE_HR = re.compile(r"\b(?P<val>\d{2,3})\s*(?P<unit>bpm|beats\/min|beats\/minute)\b", re.IGNORECASE)

_RE_VAL_UNIT = re.compile(
    r"(?P<val>(?:\d+(?:\.\d+)?)|(?:\d+\s*[–-]\s*\d+(?:\.\d+)?))\s*"
    r"(?P<unit>mmol\/l|mg\/dl|ms|s|sec|seconds|minutes|min|hours|h|days|day|weeks|week|months|month|years|year|%|mmhg)\b",
    re.IGNORECASE
)

_RE_RANGE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*[–-]\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<unit>mmol\/l|mg\/dl|ms|%|mmhg)\b",
    re.IGNORECASE
)

_ANCHOR_SETS = {
    "ECG": [r"\bqtc\b", r"\bqt\b", r"\bbazett\b", r"\belectrocardiogram\b", r"\becg\b"],
    "BP": [r"\bblood pressure\b", r"\bmmhg\b", r"\bhypertension\b", r"\bhypotension\b"],
    "HR": [r"\bheart rate\b", r"\bbeats\/min\b", r"\bbeats\/minute\b", r"\bbpm\b"],
    "CALCIUM": [r"\bionized calcium\b", r"\bserum calcium\b", r"\bca2\+\b", r"\bcalcium\b"],
    "MAGNESIUM": [r"\bmagnesium\b", r"\bmg2\+\b"],
    "EF": [r"\bejection fraction\b", r"\bef\b", r"\blvef\b"],
    "FREQ": [r"\bevery\b", r"\bper day\b", r"\bper month\b", r"\btimes\b", r"\bepisodes\b"],
    "DUR": [r"\bover the past\b", r"\bfor the previous\b", r"\blast(?:ing|ed)\b", r"\bduration\b"],
}

def find_anchors(ctx: str) -> List[str]:
    if not ctx:
        return []
    hits: List[str] = []

    def add_if(patterns: List[str], label: str) -> None:
        for pat in patterns:
            if re.search(pat, ctx, flags=re.IGNORECASE):
                hits.append(label)
                return

    for k in ["ECG", "BP", "HR", "EF", "CALCIUM", "MAGNESIUM", "FREQ", "DUR"]:
        add_if(_ANCHOR_SETS[k], k)

    lit_keep = [
        "over the past", "for the previous", "every", "per day", "per month", "lasting",
        "blood pressure", "heart rate", "ejection fraction", "bazett", "electrocardiogram", "ionized calcium",
    ]
    lctx = ctx.lower()
    for lit in lit_keep:
        if lit in lctx:
            hits.append(lit)

    out: List[str] = []
    seen = set()
    for h in hits:
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out

def _is_temporal_unit(unit_l: str) -> bool:
    return unit_l in ("days", "day", "weeks", "week", "months", "month", "years", "year", "hours", "h",
                      "minutes", "minute", "min", "seconds", "sec", "s")

def _is_lab_or_physio_unit(unit_l: str) -> bool:
    return unit_l in ("mmol/l", "mg/dl", "ms", "mmhg", "%", "bpm", "beats/min", "beats/minute")

def label_options_from_unit_and_anchors(unit: str, anchors: List[str], ctx: str) -> List[str]:
    unit_l = (unit or "").lower()
    opts: List[str] = []

    if unit_l == "ms":
        opts += ["QTc", "QT", "PR", "QRS", "OTHER"]
    elif unit_l == "mmhg":
        opts += ["BLOOD_PRESSURE", "OTHER"]
    elif unit_l in ("beats/min", "beats/minute", "bpm"):
        opts += ["HEART_RATE", "OTHER"]
    elif unit_l in ("mmol/l", "mg/dl"):
        if "CALCIUM" in anchors:
            opts += ["IONIZED_CALCIUM", "SERUM_CALCIUM", "OTHER"]
        elif "MAGNESIUM" in anchors:
            opts += ["MAGNESIUM", "OTHER"]
        else:
            opts += ["LAB_VALUE", "OTHER"]
    elif unit_l == "%":
        if "EF" in anchors:
            opts += ["LVEF", "OTHER"]
        else:
            opts += ["PERCENT_VALUE", "OTHER"]
    elif _is_temporal_unit(unit_l):
        opts += ["DURATION", "FREQUENCY", "OTHER"]
    else:
        opts += ["OTHER"]

    if _is_lab_or_physio_unit(unit_l) and not _is_temporal_unit(unit_l):
        opts += polarity_flags(ctx)

    out: List[str] = []
    seen = set()
    for x in opts:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_unit(unit: str) -> str:
    u = (unit or "").strip()
    ul = u.lower()
    if ul in ("minute", "minutes"):
        return "min"
    if ul in ("sec", "seconds"):
        return "s"
    if ul in ("day", "days"):
        return "d"
    if ul in ("year", "years"):
        return "y"
    if ul in ("month", "months"):
        return "mo"
    if ul in ("week", "weeks"):
        return "wk"
    if ul == "mmhg":
        return "mmHg"
    if ul in ("beats/minute", "beats/min"):
        return ul
    if ul == "bpm":
        return "bpm"
    if ul == "mmol/l":
        return "mmol/L"
    if ul == "mg/dl":
        return "mg/dL"
    return u

class PhaseAExtractor:
    def __init__(self, client: PubMedPMCClient):
        self.client = client

    def case_chunks_from_xml(self, pmcid: str, xml_text: str) -> List[Dict[str, Any]]:
        pmcid = _normalize_pmcid(pmcid)
        parsed = self.client.parse_pmc_xml(xml_text)
        if not parsed.get("ok"):
            raise RuntimeError(f"PMC XML parse failed: {parsed}")

        sections = parsed.get("case_sections") or []
        if not sections:
            sections = parsed.get("sections") or []

        chunks: List[Dict[str, Any]] = []

        abs_paras = parsed.get("abstract_paras") or []
        for i, p in enumerate(abs_paras[:3]):
            t = compact_ws(p)
            if t:
                chunks.append({
                    "pmcid": pmcid, "section": "Abstract", "para_idx": i, "text": t,
                    "chunk_key": f"{pmcid}|Abstract|{i}",
                })

        for sec in sections:
            sec_title = compact_ws(sec.get("title") or "UNKNOWN")
            paras = sec.get("paras") or []
            for i, p in enumerate(paras):
                t = compact_ws(p)
                if t:
                    chunks.append({
                        "pmcid": pmcid, "section": sec_title, "para_idx": i, "text": t,
                        "chunk_key": f"{pmcid}|{sec_title}|{i}",
                    })
        return chunks

    def extract_candidates_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ch in chunks:
            para = ch["text"]
            bp_spans: List[Tuple[int, int]] = []

            for m in _RE_BP.finditer(para):
                st, ed = m.span()
                bp_spans.append((st, ed))
                ctx = sentence_first_context(para, (st, ed), clause_first=True)
                anchors = find_anchors(ctx)
                out.append({
                    "type": "measurement",
                    "chunk_key": ch["chunk_key"],
                    "pmcid": ch["pmcid"],
                    "section": ch["section"],
                    "para_idx": ch["para_idx"],
                    "span_char": [st, ed],
                    "surface": para[st:ed],
                    "unit": "mmHg",
                    "value": {"systolic": float(m.group("sys")), "diastolic": float(m.group("dia"))},
                    "anchors_hit": anchors,
                    "label_options": label_options_from_unit_and_anchors("mmHg", anchors, ctx),
                    "context": ctx,
                })

            for m in _RE_HR.finditer(para):
                st, ed = m.span()
                if any((st >= bst and ed <= bed) for bst, bed in bp_spans):
                    continue
                ctx = sentence_first_context(para, (st, ed), clause_first=True)
                anchors = find_anchors(ctx)
                unit = (m.group("unit") or "").strip()
                out.append({
                    "type": "measurement",
                    "chunk_key": ch["chunk_key"],
                    "pmcid": ch["pmcid"],
                    "section": ch["section"],
                    "para_idx": ch["para_idx"],
                    "span_char": [st, ed],
                    "surface": para[st:ed],
                    "unit": normalize_unit(unit),
                    "value": float(m.group("val")),
                    "anchors_hit": anchors,
                    "label_options": label_options_from_unit_and_anchors(unit, anchors, ctx),
                    "context": ctx,
                })

            for m in _RE_VAL_UNIT.finditer(para):
                st, ed = m.span()
                if any(not (ed <= bst or st >= bed) for bst, bed in bp_spans):
                    continue

                surface = para[st:ed]
                unit = (m.group("unit") or "").strip()
                val_raw = (m.group("val") or "").strip()

                if re.search(r"[–-]", val_raw):
                    mm = re.split(r"\s*[–-]\s*", val_raw)
                    try:
                        value: Any = {"low": float(mm[0]), "high": float(mm[1])}
                    except Exception:
                        continue
                else:
                    try:
                        value = float(val_raw)
                    except Exception:
                        continue

                unit_l = unit.lower()
                clause_first = _is_lab_or_physio_unit(unit_l) and not _is_temporal_unit(unit_l)
                ctx = sentence_first_context(para, (st, ed), clause_first=clause_first)
                anchors = find_anchors(ctx)

                ref_range = None
                for rm in _RE_RANGE.finditer(ctx):
                    if (rm.group("unit") or "").lower() == unit_l:
                        try:
                            ref_range = {
                                "low": float(rm.group("lo")),
                                "high": float(rm.group("hi")),
                                "unit": normalize_unit(unit)
                            }
                            break
                        except Exception:
                            pass

                cand: Dict[str, Any] = {
                    "type": "measurement",
                    "chunk_key": ch["chunk_key"],
                    "pmcid": ch["pmcid"],
                    "section": ch["section"],
                    "para_idx": ch["para_idx"],
                    "span_char": [st, ed],
                    "surface": surface,
                    "unit": normalize_unit(unit),
                    "value": value,
                    "anchors_hit": anchors,
                    "label_options": label_options_from_unit_and_anchors(unit, anchors, ctx),
                    "context": ctx,
                }
                if ref_range:
                    cand["ref_range"] = ref_range
                out.append(cand)

        return out


# =============================================================================
# Stage3 helper
# =============================================================================

def termrecord_to_dict(r: TermRecord) -> Dict[str, Any]:
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

def load_all_seed_hpos_from_queries(queries_jsonl: str) -> List[str]:
    seeds: List[str] = []
    for obj in read_jsonl_iter(queries_jsonl):
        hid = (obj.get("hpo_id") or "").strip()
        if hid:
            seeds.append(hid)
    return list(dict.fromkeys(seeds))

def safe_esearch_pmids(client: PubMedPMCClient, term: str, retmax: int) -> List[str]:
    r = _rate_limited_call(client.pubmed_esearch, term, retmax=retmax)
    if not r.get("ok"):
        return []
    return r.get("pmids") or []

def _join_pmcids_from_pmid_map(pmids: List[str], pmid_to_pmcids: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for pmid in pmids:
        for pmcid in (pmid_to_pmcids.get(str(pmid)) or []):
            pmcid = _normalize_pmcid(pmcid)
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                out.append(pmcid)
    return out


# =============================================================================
# Cache layer: Query cache + PMCID.xml local cache
# =============================================================================

class QueryCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = (cache_dir or "").strip()
        self.path = os.path.join(self.cache_dir, "query_cache.json")
        self.data: Dict[str, Any] = {"version": 1, "queries": {}}
        self.dirty = False

        if self.cache_dir:
            ensure_dir(self.cache_dir)
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        self.data = json.load(f)
                    if "queries" not in self.data:
                        self.data = {"version": 1, "queries": {}}
            except Exception:
                self.data = {"version": 1, "queries": {}}

    def _key(self, query: str, retmax: int) -> str:
        return qkey(query, retmax)

    def get(self, query: str, retmax: int) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        k = self._key(query, retmax)
        return (self.data.get("queries") or {}).get(k)

    def set_pmids(self, query: str, retmax: int, pmids: List[str]) -> None:
        if not self.cache_dir:
            return
        k = self._key(query, retmax)
        qd = (self.data.get("queries") or {})
        if k not in qd:
            qd[k] = {"query": query, "retmax": int(retmax), "pmids": [], "pmcids": [], "ts": time.time()}
        qd[k]["pmids"] = list(pmids or [])
        qd[k]["ts"] = time.time()
        self.data["queries"] = qd
        self.dirty = True

    def set_pmcids(self, query: str, retmax: int, pmcids: List[str]) -> None:
        if not self.cache_dir:
            return
        k = self._key(query, retmax)
        qd = (self.data.get("queries") or {})
        if k not in qd:
            qd[k] = {"query": query, "retmax": int(retmax), "pmids": [], "pmcids": [], "ts": time.time()}
        qd[k]["pmcids"] = list(pmcids or [])
        qd[k]["ts"] = time.time()
        self.data["queries"] = qd
        self.dirty = True

    def flush(self) -> None:
        if not self.cache_dir or not self.dirty:
            return
        _atomic_write_json(self.path, self.data)
        self.dirty = False


class PMCXMLEntryCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = (cache_dir or "").strip()
        self.xml_dir = os.path.join(self.cache_dir, "pmc_xml") if self.cache_dir else ""
        if self.xml_dir:
            ensure_dir(self.xml_dir)

    def path_of(self, pmcid: str) -> str:
        pmcid = _normalize_pmcid(pmcid)
        return os.path.join(self.xml_dir, f"{pmcid}.xml")

    def get(self, pmcid: str) -> Optional[str]:
        if not self.xml_dir:
            return None
        p = self.path_of(pmcid)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def set(self, pmcid: str, xml_text: str) -> None:
        if not self.xml_dir:
            return
        pmcid = _normalize_pmcid(pmcid)
        p = self.path_of(pmcid)
        tmp = p + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(xml_text or "")
            os.replace(tmp, p)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


# =============================================================================
# PhaseA runner unit with DONE marker (resume key)
# =============================================================================

def per_pmcid_dir(out_phaseA_dir: str, pmcid: str) -> str:
    pmcid = _normalize_pmcid(pmcid)
    return os.path.join(out_phaseA_dir, "per_pmcid", pmcid)

def done_flag_path(out_phaseA_dir: str, pmcid: str) -> str:
    return os.path.join(per_pmcid_dir(out_phaseA_dir, pmcid), "DONE")

def is_phaseA_done(out_phaseA_dir: str, pmcid: str) -> bool:
    return os.path.exists(done_flag_path(out_phaseA_dir, pmcid))

def mark_phaseA_done(out_phaseA_dir: str, pmcid: str, meta: Dict[str, Any]) -> None:
    p = done_flag_path(out_phaseA_dir, pmcid)
    ensure_dir(os.path.dirname(p))
    _atomic_write_json(p, meta)

def run_one_pmcid_extract_from_xml(
    pmcid: str,
    xml_text: str,
    cfg: NCBIConfig,
    out_phaseA_dir: str,
) -> Tuple[str, int, int, bool]:
    """
    Returns (pmcid, n_chunks, n_cands, skipped)
    """
    pmcid = _normalize_pmcid(pmcid)
    per_dir = per_pmcid_dir(out_phaseA_dir, pmcid)
    ensure_dir(per_dir)

    if is_phaseA_done(out_phaseA_dir, pmcid):
        return pmcid, 0, 0, True

    if not (xml_text or "").strip():
        raise RuntimeError("empty xml_text (efetch/cache failed)")

    client = PubMedPMCClient(cfg)
    extractor = PhaseAExtractor(client)

    chunks = extractor.case_chunks_from_xml(pmcid, xml_text)
    cands = extractor.extract_candidates_from_chunks(chunks)

    write_jsonl(os.path.join(per_dir, "case_chunks.jsonl"), chunks)
    write_jsonl(os.path.join(per_dir, "mentions_candidates.jsonl"), cands)

    mark_phaseA_done(out_phaseA_dir, pmcid, {
        "pmcid": pmcid,
        "ts": time.time(),
        "n_chunks": len(chunks),
        "n_candidates": len(cands),
    })
    return pmcid, len(chunks), len(cands), False


# =============================================================================
# Resume helpers
# =============================================================================

def load_done_queries(query_to_pmcids_path: str, pmids_retmax: int) -> Set[str]:
    """
    Return set of qkeys (sha1(query|retmax)) that already have pmcids row.
    We treat that as "query completed".
    """
    done: Set[str] = set()
    for obj in read_jsonl_iter(query_to_pmcids_path):
        q = (obj.get("used_query") or "").strip()
        if not q:
            continue
        done.add(qkey(q, pmids_retmax))
    return done

def load_query_to_pmids_existing(query_to_pmids_path: str) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for obj in read_jsonl_iter(query_to_pmids_path):
        q = (obj.get("used_query") or "").strip()
        pmids = obj.get("pmids") or []
        if q and isinstance(pmids, list):
            m[q] = [str(x) for x in pmids if str(x).strip()]
    return m

def load_query_to_pmcids_existing(query_to_pmcids_path: str) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for obj in read_jsonl_iter(query_to_pmcids_path):
        q = (obj.get("used_query") or "").strip()
        pmcids = obj.get("pmcids") or []
        if q and isinstance(pmcids, list):
            m[q] = [_normalize_pmcid(x) for x in pmcids if x]
    return m

def load_done_pmcids_from_done(out_phaseA_dir: str) -> Set[str]:
    done: Set[str] = set()
    base = os.path.join(out_phaseA_dir, "per_pmcid")
    if not os.path.isdir(base):
        return done
    for name in os.listdir(base):
        pmcid = _normalize_pmcid(name)
        if not pmcid:
            continue
        if is_phaseA_done(out_phaseA_dir, pmcid):
            done.add(pmcid)
    return done


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser("run_stage3_phaseA_full")

    ap.add_argument("--queries_jsonl", type=str, required=True)
    ap.add_argument("--neighbors_jsonl", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True, help="Base out dir (will create a run_id subdir)")
    ap.add_argument("--run_name", type=str, default="", help="Optional run name suffix")

    ap.add_argument("--resume_dir", type=str, default="", help="Existing run_dir to resume (skip finished work)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_seed0", type=int, default=1)
    ap.add_argument("--n_seed1", type=int, default=1)
    ap.add_argument("--n_seed2", type=int, default=1)
    ap.add_argument("--n_neg", type=int, default=1)
    ap.add_argument("--max_seeds", type=int, default=0, help="0 means ALL seeds. Otherwise limit to first N seeds.")

    ap.add_argument("--email", type=str, required=True)
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--tool", type=str, default="hpo-agent-stage3-full")
    ap.add_argument("--polite_sleep", type=float, default=0.34, help="Global spacing across ALL threads")
    ap.add_argument("--threads", type=int, default=3, help="PhaseA parse/extract worker threads (recommend 2~4)")

    ap.add_argument("--check_esearch", action="store_true", help="Let sampler compute hits and allow 0-hit trigger repair")
    ap.add_argument("--use_llm_repair", action="store_true")
    ap.add_argument("--llm_model", type=str, default="deepseek-chat")
    ap.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--llm_timeout", type=float, default=60.0)

    ap.add_argument("--pmids_retmax", type=int, default=50, help="PMIDs retrieved per query")
    ap.add_argument("--pmc_per_query", type=int, default=8, help="Cap PMCIDs per query to avoid explosion (0=unlimited)")

    ap.add_argument("--cache_dir", type=str, default="", help="Cache dir. Default: <run_dir>/.cache")
    ap.add_argument("--flush_every", type=int, default=50, help="Flush query outputs/caches every N unique queries")
    ap.add_argument("--elink_batch_size", type=int, default=200, help="Batch size for PubMed->PMC ELink on unique PMIDs")
    ap.add_argument("--efetch_batch_size", type=int, default=20, help="Batch size for PMC EFetch on unique PMCIDs")

    args = ap.parse_args()

    # -----------------------------
    # Create or resume run directory
    # -----------------------------
    if args.resume_dir.strip():
        run_dir = args.resume_dir.strip()
        if not os.path.isdir(run_dir):
            raise RuntimeError(f"--resume_dir not found: {run_dir}")
        run_id = os.path.basename(run_dir.rstrip("/"))
        print(f"[RESUME] run_dir={run_dir}")
    else:
        run_id = now_stamp()
        if args.run_name.strip():
            run_id = f"{run_id}_{args.run_name.strip()}"
        run_dir = os.path.join(args.out_dir, run_id)
        ensure_dir(run_dir)
        print(f"[NEW RUN] run_dir={run_dir}")

    # layout
    dir_terms = os.path.join(run_dir, "terms")
    dir_pubmed = os.path.join(run_dir, "pubmed")
    dir_phaseA = os.path.join(run_dir, "phaseA")
    dir_phaseA_merged = os.path.join(dir_phaseA, "merged")
    dir_metrics = os.path.join(run_dir, "metrics")
    for d in (dir_terms, dir_pubmed, dir_phaseA, dir_phaseA_merged, dir_metrics):
        ensure_dir(d)

    failures_path = os.path.join(dir_metrics, "failures.jsonl")

    # global rate limiter
    global _RATE_LIMITER
    _RATE_LIMITER = GlobalRateLimiter(min_interval=float(args.polite_sleep))

    # NCBI config (polite_sleep=0 because we do global throttling)
    cfg = NCBIConfig(
        email=args.email,
        tool=args.tool,
        api_key=args.api_key,
        polite_sleep=0.0,
    )
    ncbi_client = PubMedPMCClient(cfg)

    # -----------------------------
    # Cache defaults to output dir (MINIMIZED CHANGE)
    # -----------------------------
    cache_dir = (args.cache_dir or "").strip()
    if not cache_dir:
        cache_dir = os.path.join(run_dir, ".cache")
    ensure_dir(cache_dir)

    qcache = QueryCache(cache_dir)
    pmc_xml_cache = PMCXMLEntryCache(cache_dir)

    # build sampler
    sampler = ClusterTermSampler(
        queries_jsonl=args.queries_jsonl,
        neighbors_jsonl=args.neighbors_jsonl,
        seed=int(args.seed),
    )

    llm = None
    if args.use_llm_repair:
        from llm_client import LLMClient  # type: ignore
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set but --use_llm_repair enabled.")
        llm = LLMClient(
            api_key=api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            timeout=float(args.llm_timeout),
        )

    # Load seeds
    seeds = load_all_seed_hpos_from_queries(args.queries_jsonl)
    if args.max_seeds and int(args.max_seeds) > 0:
        seeds = seeds[: int(args.max_seeds)]

    print(f"[INFO] n_seeds={len(seeds)} check_esearch={bool(args.check_esearch)} use_llm_repair={bool(args.use_llm_repair)} cache_dir={cache_dir}")

    # -----------------------------
    # Part 1: Stage3 terms (optional resume)
    # -----------------------------
    terms_path = os.path.join(dir_terms, "terms.jsonl")
    selected_path = os.path.join(dir_terms, "terms_selected.jsonl")

    if os.path.exists(selected_path):
        terms_selected = list(read_jsonl_iter(selected_path))
        print(f"[RESUME] reuse terms_selected.jsonl: {len(terms_selected)}")
    else:
        terms_all: List[TermRecord] = []
        pbar_seeds = tqdm(seeds, desc="Stage3: sampling terms over seeds", dynamic_ncols=True)
        for seed_hpo in pbar_seeds:
            for _ in range(max(0, int(args.n_seed0))):
                r = sampler.build_seed0(seed_hpo)
                if r:
                    r = sampler.maybe_repair(r, llm=llm, client=ncbi_client, check_esearch=bool(args.check_esearch))
                    terms_all.append(r)
            for _ in range(max(0, int(args.n_seed1))):
                r = sampler.build_seed1(seed_hpo)
                if r:
                    r = sampler.maybe_repair(r, llm=llm, client=ncbi_client, check_esearch=bool(args.check_esearch))
                    terms_all.append(r)
            for _ in range(max(0, int(args.n_seed2))):
                r = sampler.build_seed2(seed_hpo)
                if r:
                    r = sampler.maybe_repair(r, llm=llm, client=ncbi_client, check_esearch=bool(args.check_esearch))
                    terms_all.append(r)
            for _ in range(max(0, int(args.n_neg))):
                r = sampler.build_neg(seed_hpo)
                if r:
                    r = sampler.maybe_repair(r, llm=llm, client=ncbi_client, check_esearch=bool(args.check_esearch))
                    terms_all.append(r)

        write_jsonl(terms_path, (termrecord_to_dict(x) for x in terms_all))

        terms_selected: List[Dict[str, Any]] = []
        for r in terms_all:
            d = termrecord_to_dict(r)
            uh = d.get("used_hits")
            if args.check_esearch:
                if isinstance(uh, int) and uh > 0:
                    terms_selected.append(d)
            else:
                terms_selected.append(d)

        write_jsonl(selected_path, terms_selected)
        print(f"[INFO] wrote terms_all={len(terms_all)} terms_selected={len(terms_selected)}")

    # -----------------------------
    # Part 2: Query -> PMIDs/PMCIDs (RESUME SAFE)
    # -----------------------------
    query_to_pmids_path = os.path.join(dir_pubmed, "query_to_pmids.jsonl")
    query_to_pmcids_path = os.path.join(dir_pubmed, "query_to_pmcids.jsonl")

    # Build query_metas from selected terms
    query_metas: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in terms_selected:
        used_query = (rec.get("used_query") or rec.get("query") or "").strip()
        if not used_query:
            continue
        query_metas[used_query].append({
            "seed_hpo": rec.get("seed_hpo"),
            "kind": rec.get("kind"),
            "used_query": used_query,
        })
    unique_queries: List[str] = list(query_metas.keys())

    # Resume: load already done queries
    done_qkeys: Set[str] = set()
    if os.path.exists(query_to_pmcids_path):
        done_qkeys = load_done_queries(query_to_pmcids_path, int(args.pmids_retmax))
        print(f"[RESUME] done_queries={len(done_qkeys)} (from query_to_pmcids.jsonl)")

    # Also load existing mappings if present (so later steps can reuse)
    query_to_pmids_existing = load_query_to_pmids_existing(query_to_pmids_path) if os.path.exists(query_to_pmids_path) else {}
    query_to_pmcids_existing = load_query_to_pmcids_existing(query_to_pmcids_path) if os.path.exists(query_to_pmcids_path) else {}

    # ---- Step 2.1 ESearch unique queries (skip done) ----
    query_to_pmids: Dict[str, List[str]] = dict(query_to_pmids_existing)
    unique_pmids_set: Set[str] = set()

    for pmids in query_to_pmids.values():
        for pmid in pmids:
            unique_pmids_set.add(str(pmid))

    todo_queries = [q for q in unique_queries if qkey(q, int(args.pmids_retmax)) not in done_qkeys]
    pbar_q1 = tqdm(todo_queries, desc="Stage3: ESearch todo queries", dynamic_ncols=True)

    buf_pmids_rows: List[Dict[str, Any]] = []
    for i, q in enumerate(pbar_q1, start=1):
        try:
            cached = qcache.get(q, int(args.pmids_retmax))
            if cached and isinstance(cached.get("pmids"), list) and cached.get("retmax") == int(args.pmids_retmax):
                pmids = [str(x) for x in (cached.get("pmids") or []) if str(x).strip()]
            else:
                pmids = safe_esearch_pmids(ncbi_client, q, retmax=int(args.pmids_retmax))
                qcache.set_pmids(q, int(args.pmids_retmax), pmids)

            pmids = [str(x) for x in pmids if str(x).strip()]
            query_to_pmids[q] = pmids
            for pmid in pmids:
                unique_pmids_set.add(str(pmid))

            meta0 = query_metas[q][0] if query_metas[q] else {"seed_hpo": None, "kind": None, "used_query": q}
            buf_pmids_rows.append({
                "seed_hpo": meta0.get("seed_hpo"),
                "kind": meta0.get("kind"),
                "used_query": q,
                "pmids_retmax": int(args.pmids_retmax),
                "pmids": pmids,
            })

        except Exception as e:
            append_jsonl(failures_path, [{
                "stage": "pubmed_esearch",
                "used_query": q,
                "error": str(e),
            }])

        if int(args.flush_every) > 0 and (i % int(args.flush_every) == 0):
            if buf_pmids_rows:
                append_jsonl(query_to_pmids_path, buf_pmids_rows)
                buf_pmids_rows = []
            qcache.flush()

    if buf_pmids_rows:
        append_jsonl(query_to_pmids_path, buf_pmids_rows)
    qcache.flush()

    unique_pmids: List[str] = sorted(unique_pmids_set)

    # ---- Step 2.2 Global batch ELink on unique PMIDs ----
    pmid_to_pmcids: Dict[str, List[str]] = {}
    if unique_pmids:
        pbar_elink = tqdm(total=len(unique_pmids), desc="Stage3: ELink unique PMIDs (batch)", dynamic_ncols=True)
        bs = max(1, int(args.elink_batch_size))
        for chunk_start in range(0, len(unique_pmids), bs):
            chunk = unique_pmids[chunk_start:chunk_start + bs]
            try:
                r = _rate_limited_call(ncbi_client.pubmed_elink_to_pmc_batch, chunk, batch_size=bs)
                if not r.get("ok"):
                    append_jsonl(failures_path, [{
                        "stage": "pubmed_elink_batch",
                        "error": r.get("error", ""),
                        "text": r.get("text", ""),
                        "failed_batch_size": len(chunk),
                    }])
                else:
                    mp2 = r.get("mapping") or {}
                    for pmid, pmc_list in mp2.items():
                        pmid_to_pmcids[str(pmid)] = [_normalize_pmcid(x) for x in (pmc_list or []) if x]
            except Exception as e:
                append_jsonl(failures_path, [{
                    "stage": "pubmed_elink_batch",
                    "error": str(e),
                    "failed_batch_size": len(chunk),
                }])
            finally:
                pbar_elink.update(len(chunk))
        pbar_elink.close()

    # ---- Step 2.3 Build PMCIDs per query (skip already done) ----
    query_to_pmcids: Dict[str, List[str]] = dict(query_to_pmcids_existing)
    buf_pmcids_rows: List[Dict[str, Any]] = []

    pmcid_index_path = os.path.join(dir_metrics, "pmcid_index.jsonl")
    pmcid_meta_seen: Set[str] = set()
    if os.path.exists(pmcid_index_path):
        for obj in read_jsonl_iter(pmcid_index_path):
            pmcid = _normalize_pmcid(obj.get("pmcid") or "")
            if pmcid:
                pmcid_meta_seen.add(pmcid)

    all_pmcids: List[str] = []
    seen_pmcids: Set[str] = set()

    for pmcids in query_to_pmcids.values():
        for pmcid in pmcids:
            pmcid = _normalize_pmcid(pmcid)
            if pmcid and pmcid not in seen_pmcids:
                seen_pmcids.add(pmcid)
                all_pmcids.append(pmcid)

    pbar_q2 = tqdm(unique_queries, desc="Stage3: build PMCIDs per query (resume-safe)", dynamic_ncols=True)
    new_pmcid_meta_rows: List[Dict[str, Any]] = []

    for j, q in enumerate(pbar_q2, start=1):
        k = qkey(q, int(args.pmids_retmax))
        if k in done_qkeys:
            continue

        pmids = query_to_pmids.get(q, [])
        pmcids = _join_pmcids_from_pmid_map(pmids, pmid_to_pmcids)

        if args.pmc_per_query and int(args.pmc_per_query) > 0:
            pmcids = pmcids[: int(args.pmc_per_query)]

        qcache.set_pmcids(q, int(args.pmids_retmax), pmcids)

        meta0 = query_metas[q][0] if query_metas[q] else {"seed_hpo": None, "kind": None, "used_query": q}

        buf_pmcids_rows.append({
            "seed_hpo": meta0.get("seed_hpo"),
            "kind": meta0.get("kind"),
            "used_query": q,
            "pmcids": pmcids,
        })

        query_to_pmcids[q] = pmcids

        for pmcid in pmcids:
            pmcid = _normalize_pmcid(pmcid)
            if not pmcid:
                continue
            if pmcid not in seen_pmcids:
                seen_pmcids.add(pmcid)
                all_pmcids.append(pmcid)

            if pmcid not in pmcid_meta_seen:
                pmcid_meta_seen.add(pmcid)
                new_pmcid_meta_rows.append({
                    "pmcid": pmcid,
                    "from_seed_hpo": meta0.get("seed_hpo"),
                    "kind": meta0.get("kind"),
                    "used_query": q,
                })

        if int(args.flush_every) > 0 and (j % int(args.flush_every) == 0):
            if buf_pmcids_rows:
                append_jsonl(query_to_pmcids_path, buf_pmcids_rows)
                buf_pmcids_rows = []
            if new_pmcid_meta_rows:
                append_jsonl(pmcid_index_path, new_pmcid_meta_rows)
                new_pmcid_meta_rows = []
            qcache.flush()

    if buf_pmcids_rows:
        append_jsonl(query_to_pmcids_path, buf_pmcids_rows)
    if new_pmcid_meta_rows:
        append_jsonl(pmcid_index_path, new_pmcid_meta_rows)
    qcache.flush()

    # -----------------------------
    # Part 3: PMC EFetch + PhaseA (RESUME SAFE)
    # -----------------------------
    merged_case = os.path.join(dir_phaseA_merged, "case_chunks.jsonl")
    merged_mentions = os.path.join(dir_phaseA_merged, "mentions_candidates.jsonl")
    ensure_dir(dir_phaseA_merged)

    done_pmcids = load_done_pmcids_from_done(dir_phaseA)
    todo_pmcids = [p for p in all_pmcids if _normalize_pmcid(p) not in done_pmcids]
    todo_pmcids = [_normalize_pmcid(x) for x in todo_pmcids if _normalize_pmcid(x)]

    print(f"[RESUME] phaseA done_pmcids={len(done_pmcids)}  todo_pmcids={len(todo_pmcids)}  all_pmcids={len(all_pmcids)}")

    pmcid_to_xml: Dict[str, str] = {}
    missing_pmcids: List[str] = []

    for pmcid in todo_pmcids:
        xml_hit = pmc_xml_cache.get(pmcid)
        if xml_hit:
            pmcid_to_xml[pmcid] = xml_hit
        else:
            missing_pmcids.append(pmcid)

    if missing_pmcids:
        pbar_fetch = tqdm(total=len(missing_pmcids), desc="Stage3: PMC EFetch missing PMCIDs (batch)", dynamic_ncols=True)
        bs = max(1, int(args.efetch_batch_size))
        for chunk_start in range(0, len(missing_pmcids), bs):
            chunk = missing_pmcids[chunk_start:chunk_start + bs]
            try:
                rr = _rate_limited_call(ncbi_client.pmc_efetch_xml_batch, chunk, batch_size=bs)
                if not rr.get("ok"):
                    append_jsonl(failures_path, [{
                        "stage": "pmc_efetch_batch",
                        "error": rr.get("error", ""),
                        "text": rr.get("text", ""),
                        "failed_batch_size": len(chunk),
                    }])
                else:
                    mp3 = rr.get("pmcid_to_xml") or {}
                    for pmcid, xml_text in mp3.items():
                        pmcid = _normalize_pmcid(pmcid)
                        if pmcid and xml_text:
                            pmcid_to_xml[pmcid] = xml_text
                            pmc_xml_cache.set(pmcid, xml_text)
            except Exception as e:
                append_jsonl(failures_path, [{
                    "stage": "pmc_efetch_batch",
                    "error": str(e),
                    "failed_batch_size": len(chunk),
                }])
            finally:
                pbar_fetch.update(len(chunk))
        pbar_fetch.close()

    lock_merge = threading.Lock()
    ok, fail, skipped = 0, 0, 0
    total_chunks, total_cands = 0, 0

    pbar_extract = tqdm(total=len(todo_pmcids), desc="Stage3: PhaseA parse/extract PMCIDs (resume-safe)", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=int(args.threads)) as ex:
        futs = {}
        for pmcid in todo_pmcids:
            xml_text = pmcid_to_xml.get(pmcid, "")
            futs[ex.submit(run_one_pmcid_extract_from_xml, pmcid, xml_text, cfg, dir_phaseA)] = pmcid

        for fut in as_completed(futs):
            pmcid = futs[fut]
            try:
                pmcid, n_chunks, n_cands, was_skipped = fut.result()
                if was_skipped:
                    skipped += 1
                    continue

                total_chunks += int(n_chunks)
                total_cands += int(n_cands)

                per_dir = per_pmcid_dir(dir_phaseA, pmcid)
                per_case = os.path.join(per_dir, "case_chunks.jsonl")
                per_mentions = os.path.join(per_dir, "mentions_candidates.jsonl")

                with open(per_case, "r", encoding="utf-8") as f1, open(per_mentions, "r", encoding="utf-8") as f2:
                    case_lines = f1.read().splitlines()
                    mention_lines = f2.read().splitlines()

                with lock_merge:
                    with open(merged_case, "a", encoding="utf-8") as f:
                        for ln in case_lines:
                            f.write(ln + "\n")
                    with open(merged_mentions, "a", encoding="utf-8") as f:
                        for ln in mention_lines:
                            f.write(ln + "\n")

                ok += 1
            except Exception as e:
                fail += 1
                append_jsonl(failures_path, [{
                    "stage": "phaseA_extract",
                    "pmcid": pmcid,
                    "error": str(e),
                }])
            finally:
                pbar_extract.update(1)
    pbar_extract.close()

    summary = {
        "run_dir": run_dir,
        "time": run_id,
        "resume_dir": args.resume_dir.strip() or None,
        "n_seeds": len(seeds),

        "sampler": {
            "check_esearch": bool(args.check_esearch),
            "use_llm_repair": bool(args.use_llm_repair),
            "terms_selected_path": os.path.join(dir_terms, "terms_selected.jsonl"),
        },

        "pubmed": {
            "pmids_retmax": int(args.pmids_retmax),
            "pmc_per_query": int(args.pmc_per_query),
            "n_unique_queries": len(unique_queries),
            "n_done_queries": len(done_qkeys),
            "n_todo_queries": len(todo_queries),
            "n_unique_pmids": len(unique_pmids),
            "n_unique_pmcids": len(all_pmcids),
            "elink_batch_size": int(args.elink_batch_size),
        },

        "pmc": {
            "efetch_batch_size": int(args.efetch_batch_size),
            "cache_dir": cache_dir,
        },

        "phaseA": {
            "threads": int(args.threads),
            "polite_sleep": float(args.polite_sleep),
            "done_pmcids": len(done_pmcids),
            "todo_pmcids": len(todo_pmcids),
            "ok_pmcids": ok,
            "fail_pmcids": fail,
            "skipped_pmcids": skipped,
            "total_chunks_new": total_chunks,
            "total_candidates_new": total_cands,
            "merged_case_chunks_jsonl": merged_case,
            "merged_mentions_candidates_jsonl": merged_mentions,
        },

        "paths": {
            "terms_selected": os.path.join(dir_terms, "terms_selected.jsonl"),
            "query_to_pmids": query_to_pmids_path,
            "query_to_pmcids": query_to_pmcids_path,
            "pmcid_index": os.path.join(dir_metrics, "pmcid_index.jsonl"),
            "failures": failures_path,
            "summary": os.path.join(dir_metrics, "summary.json"),
        },
    }

    _atomic_write_json(os.path.join(dir_metrics, "summary.json"), summary)

    print("\n[DONE] Full Stage3 run finished (resume-safe).")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
