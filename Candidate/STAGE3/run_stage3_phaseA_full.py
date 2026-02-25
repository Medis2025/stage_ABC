#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_stage3_phaseA_full.py  (RESUME-SAFE + HPO indexes + evidence_pool + manifest/run_config)

新增能力：
1) HPO 级 index：
   - pubmed/hpo_to_pmids.jsonl
   - pubmed/hpo_to_pmcids.jsonl

2) Evidence pool：
   - pubmed/pmid_to_abstract.jsonl       (pmid -> title/abstract/metadata)
   - pubmed/evidence_pool.jsonl          (按 HPO 展开，每行：hpo_id + pmid + sha1_abstract_norm 等)
   - pubmed/by_hpo/<HPO>.jsonl           (同 evidence_pool，但按 HPO 分文件)

3) 运行配置 & 清单：
   - run_config.json                     (记录 args 与基础配置)
   - manifest.json                       (本次 run 的重要输出路径与统计)

注意：
- PhaseB 在写 corpus 时，建议使用本文件里的 `_sha1(normalized_text)` 对生成的 text 做 hash，
  并在 corpus JSONL 中增加 `sha1_text_norm` 字段。
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
# Debug helpers
# =============================================================================

def _short(s: Any, n: int = 180) -> str:
    if s is None:
        return "None"
    t = str(s)
    t = t.replace("\n", " ").replace("\r", " ")
    if len(t) <= n:
        return t
    return t[:n] + " ..."

def _head_file(path: str, n_lines: int = 3, max_chars: int = 200) -> str:
    if not path or not os.path.exists(path):
        return f"[HEAD] {path} (missing)"
    try:
        lines: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(n_lines):
                ln = f.readline()
                if not ln:
                    break
                ln = ln.rstrip("\n")
                if len(ln) > max_chars:
                    ln = ln[:max_chars] + " ..."
                lines.append(ln)
        if not lines:
            return f"[HEAD] {path} (empty)"
        return f"[HEAD] {path}\n  " + "\n  ".join(lines)
    except Exception as e:
        return f"[HEAD] {path} (read error: {e})"

def _count_jsonl_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def _append_and_flush_jsonl(path: str, row: Dict[str, Any]) -> None:
    append_jsonl(path, [row])


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
# PhaseA extractor (same as before)
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

    # NOTE: literal pass（原逻辑保留）
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
                clause_first_flag = _is_lab_or_physio_unit(unit_l) and not _is_temporal_unit(unit_l)
                ctx = sentence_first_context(para, (st, ed), clause_first=clause_first_flag)
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
# Cache layer
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


class PubMedXMLEntryCache:
    """
    Cache for PubMed XML entries (weak evidence layer).

    Files live under:
      <cache_dir>/pmid_xml/<pmid>.xml
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = (cache_dir or "").strip()
        self.xml_dir = os.path.join(self.cache_dir, "pmid_xml") if self.cache_dir else ""
        if self.xml_dir:
            ensure_dir(self.xml_dir)

    def path_of(self, pmid: str) -> str:
        pmid = (pmid or "").strip()
        return os.path.join(self.xml_dir, f"{pmid}.xml")

    def get(self, pmid: str) -> Optional[str]:
        if not self.xml_dir:
            return None
        p = self.path_of(pmid)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def set(self, pmid: str, xml_text: str) -> None:
        if not self.xml_dir:
            return
        pmid = (pmid or "").strip()
        if not pmid:
            return
        p = self.path_of(pmid)
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
) -> Tuple[str, int, int, bool, str, str]:
    """
    Returns (pmcid, n_chunks, n_cands, skipped, head_chunk_text, head_cand_surface)
    """
    pmcid = _normalize_pmcid(pmcid)
    per_dir = per_pmcid_dir(out_phaseA_dir, pmcid)
    ensure_dir(per_dir)

    if is_phaseA_done(out_phaseA_dir, pmcid):
        return pmcid, 0, 0, True, "", ""

    if not (xml_text or "").strip():
        raise RuntimeError("empty xml_text (efetch/cache failed)")

    client = PubMedPMCClient(cfg)
    extractor = PhaseAExtractor(client)

    chunks = extractor.case_chunks_from_xml(pmcid, xml_text)
    cands = extractor.extract_candidates_from_chunks(chunks)

    write_jsonl(os.path.join(per_dir, "case_chunks.jsonl"), chunks)
    write_jsonl(os.path.join(per_dir, "mentions_candidates.jsonl"), cands)

    head_chunk = ""
    if chunks:
        head_chunk = _short((chunks[0].get("text") or ""), 220)
    head_cand = ""
    if cands:
        head_cand = _short((cands[0].get("surface") or ""), 120)

    mark_phaseA_done(out_phaseA_dir, pmcid, {
        "pmcid": pmcid,
        "ts": time.time(),
        "n_chunks": len(chunks),
        "n_candidates": len(cands),
        "head_chunk": head_chunk,
        "head_candidate_surface": head_cand,
    })
    return pmcid, len(chunks), len(cands), False, head_chunk, head_cand


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
        done.add(qkey(q, retmax=pmids_retmax))
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
# NEW: HPO-level index & evidence_pool builders
# =============================================================================

def _write_hpo_index_jsonl(path: str, mapping: Dict[str, Set[str]], key_name: str) -> int:
    """
    mapping: hpo_id -> set(ids)
    key_name: 'pmids' or 'pmcids'
    Returns: number of HPO rows written
    """
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for hpo_id in sorted(mapping.keys()):
            ids = sorted(mapping[hpo_id])
            if not ids:
                continue
            row = {"hpo_id": hpo_id, key_name: ids}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n

def build_and_write_hpo_indexes(
    pmid_to_hpos: DefaultDict[str, Set[str]],
    pmcid_to_hpos: DefaultDict[str, Set[str]],
    hpo_to_pmids_path: str,
    hpo_to_pmcids_path: str,
) -> Tuple[int, int]:
    """
    根据 pmid_to_hpos / pmcid_to_hpos 构建：
      - hpo_to_pmids.jsonl
      - hpo_to_pmcids.jsonl
    返回：(num_hpo_in_pmids_index, num_hpo_in_pmcids_index)
    """
    hpo_to_pmids: DefaultDict[str, Set[str]] = defaultdict(set)
    for pmid, hpos in tqdm(
        pmid_to_hpos.items(),
        desc="Stage3: build hpo_to_pmids index",
        dynamic_ncols=True
    ):
        pmid_str = str(pmid).strip()
        if not pmid_str:
            continue
        for h in hpos:
            h = (h or "").strip()
            if h:
                hpo_to_pmids[h].add(pmid_str)

    hpo_to_pmcids: DefaultDict[str, Set[str]] = defaultdict(set)
    for pmcid, hpos in tqdm(
        pmcid_to_hpos.items(),
        desc="Stage3: build hpo_to_pmcids index",
        dynamic_ncols=True
    ):
        pmcid_norm = _normalize_pmcid(pmcid)
        if not pmcid_norm:
            continue
        for h in hpos:
            h = (h or "").strip()
            if h:
                hpo_to_pmcids[h].add(pmcid_norm)

    n_pmids = _write_hpo_index_jsonl(hpo_to_pmids_path, hpo_to_pmids, "pmids")
    n_pmcids = _write_hpo_index_jsonl(hpo_to_pmcids_path, hpo_to_pmcids, "pmcids")
    return n_pmids, n_pmcids

def build_evidence_pool(
    pmid_to_abstract_path: str,
    pmid_to_hpos: DefaultDict[str, Set[str]],
    pubmed_dir: str,
) -> str:
    """
    构建 evidence_pool：
      - pubmed/evidence_pool.jsonl （每行：hpo_id + pmid + abstract + sha1_abstract_norm 等）
      - pubmed/by_hpo/<HPO>.jsonl （按 HPO 切分的视图）

    注意：
      - 对 abstract 做 normalized（compact_ws）后，再 sha1
      - 只对有 HPO 映射的 pmid 输出（pmid_to_hpos）
    """
    ensure_dir(pubmed_dir)
    by_hpo_dir = os.path.join(pubmed_dir, "by_hpo")
    ensure_dir(by_hpo_dir)

    # 清空旧的 by_hpo/*（防止残留）
    for fn in os.listdir(by_hpo_dir):
        fp = os.path.join(by_hpo_dir, fn)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception:
            pass

    evidence_pool_path = os.path.join(pubmed_dir, "evidence_pool.jsonl")
    tmp_pool = evidence_pool_path + ".tmp"

    hpo_fhs: Dict[str, Any] = {}
    n_rows = 0
    try:
        total_abs_lines = _count_jsonl_lines(pmid_to_abstract_path)
        with open(tmp_pool, "w", encoding="utf-8") as f_pool:
            iter_objs = read_jsonl_iter(pmid_to_abstract_path)
            if total_abs_lines > 0:
                iter_objs = tqdm(
                    iter_objs,
                    total=total_abs_lines,
                    desc="Stage3: build evidence_pool",
                    dynamic_ncols=True,
                )
            for obj in iter_objs:
                pmid = str(obj.get("pmid") or "").strip()
                if not pmid:
                    continue
                hpos = sorted(pmid_to_hpos.get(pmid, []))
                if not hpos:
                    continue

                title = obj.get("title") or ""
                abstract = obj.get("abstract") or ""
                journal = obj.get("journal") or ""
                year = obj.get("year")

                norm_abs = compact_ws(abstract)
                sha_abs = _sha1(norm_abs)

                try:
                    year_int = int(year) if year is not None and str(year).isdigit() else None
                except Exception:
                    year_int = None

                base_rec = {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "year_int": year_int,
                    "has_abstract": bool(abstract),
                    "len_abstract_norm": len(norm_abs),
                    "sha1_abstract_norm": sha_abs,
                }

                for h in hpos:
                    h = (h or "").strip()
                    if not h:
                        continue
                    rec = dict(base_rec)
                    rec["hpo_id"] = h
                    line = json.dumps(rec, ensure_ascii=False)

                    # 写入全局 evidence_pool
                    f_pool.write(line + "\n")
                    n_rows += 1

                    # 写入 per-HPO 视图
                    if h not in hpo_fhs:
                        h_fp = os.path.join(by_hpo_dir, f"{h}.jsonl")
                        hpo_fhs[h] = open(h_fp + ".tmp", "w", encoding="utf-8")
                    hpo_fhs[h].write(line + "\n")
    finally:
        # flush & close per-HPO files and rename
        for h, fh in hpo_fhs.items():
            try:
                fh.close()
                tmp_fp = os.path.join(by_hpo_dir, f"{h}.jsonl.tmp")
                final_fp = os.path.join(by_hpo_dir, f"{h}.jsonl")
                os.replace(tmp_fp, final_fp)
            except Exception:
                pass

    os.replace(tmp_pool, evidence_pool_path)
    print(f"[INFO] evidence_pool rows={n_rows} path={evidence_pool_path}")
    return evidence_pool_path


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

    ap.add_argument("--check_esearch", action="store_true", help="(Stage1) kept for backward compatibility; no longer used for sampling")
    ap.add_argument("--use_llm_repair", action="store_true")
    ap.add_argument("--llm_model", type=str, default="deepseek-chat")
    ap.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--llm_timeout", type=float, default=60.0)

    ap.add_argument("--pmids_retmax", type=int, default=50, help="PMIDs retrieved per query")
    ap.add_argument("--pmc_per_query", type=int, default=8, help="Cap PMCIDs per query to avoid explosion (0=unlimited)")

    ap.add_argument("--cache_dir", type=str, default="", help="Cache dir for PMC XML. Default: <run_dir>/.cache")
    ap.add_argument("--flush_every", type=int, default=50, help="(kept) still used for periodic messages")
    ap.add_argument("--elink_batch_size", type=int, default=200, help="Batch size for PubMed->PMC ELink on unique PMIDs")
    ap.add_argument("--efetch_batch_size", type=int, default=20, help="Batch size for PMC EFetch on unique PMCIDs")

    ap.add_argument("--debug", action="store_true", help="Print debug info (heads, null checks).")
    ap.add_argument("--debug_every", type=int, default=50, help="Print debug every N items for long loops.")
    ap.add_argument("--head_lines", type=int, default=2, help="How many head lines to show for key files.")
    ap.add_argument("--head_chars", type=int, default=220, help="Max chars per head line.")

    args = ap.parse_args()

    def dprint(msg: str) -> None:
        if args.debug:
            print(msg, flush=True)

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

    # NEW: run_config（记录 args 等）
    run_config = {
        "run_id": run_id,
        "time_start": now_stamp(),
        "args": vars(args),
    }
    _atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)

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

    # PMC cache defaults to run_dir/.cache
    cache_dir = (args.cache_dir or "").strip()
    if not cache_dir:
        cache_dir = os.path.join(run_dir, ".cache")
    ensure_dir(cache_dir)
    pmc_xml_cache = PMCXMLEntryCache(cache_dir)

    # PubMed cache is always under run_dir/pubmed/
    pubmed_cache_dir = dir_pubmed
    ensure_dir(pubmed_cache_dir)
    qcache = QueryCache(pubmed_cache_dir)
    pubmed_xml_cache = PubMedXMLEntryCache(pubmed_cache_dir)

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

    print(f"[INFO] n_seeds={len(seeds)} check_esearch={bool(args.check_esearch)} use_llm_repair={bool(args.use_llm_repair)} pubmed_cache_dir={pubmed_cache_dir} pmc_cache_dir={cache_dir}")

    # debug: input head
    if args.debug:
        dprint(_head_file(args.queries_jsonl, n_lines=args.head_lines, max_chars=args.head_chars))
        dprint(_head_file(args.neighbors_jsonl, n_lines=args.head_lines, max_chars=args.head_chars))

    # -----------------------------
    # Part 1: Stage3 terms (RESUME-SAFE on terms.jsonl) — purely local now
    # -----------------------------
    terms_path = os.path.join(dir_terms, "terms.jsonl")
    selected_path = os.path.join(dir_terms, "terms_selected.jsonl")

    if os.path.exists(selected_path):
        terms_selected = list(read_jsonl_iter(selected_path))
        print(f"[RESUME] reuse terms_selected.jsonl: {len(terms_selected)}")
        if args.debug:
            dprint(_head_file(selected_path, n_lines=args.head_lines, max_chars=args.head_chars))
    else:
        terms_all: List[Dict[str, Any]] = []
        seeds_done: Set[str] = set()

        if os.path.exists(terms_path):
            for obj in read_jsonl_iter(terms_path):
                terms_all.append(obj)
                sh = (obj.get("seed_hpo") or "").strip()
                if sh:
                    seeds_done.add(sh)
            print(f"[RESUME] found existing terms.jsonl: {len(terms_all)} terms, {len(seeds_done)} seeds with prior terms")
        else:
            ensure_dir(os.path.dirname(terms_path))

        pbar_seeds = tqdm(seeds, desc="Stage3: sampling terms over seeds", dynamic_ncols=True)
        for si, seed_hpo in enumerate(pbar_seeds, start=1):
            if seeds_done and seed_hpo in seeds_done:
                if args.debug and (si == 1 or (si % max(1, int(args.debug_every)) == 0)):
                    dprint(f"[DBG][terms] skip seed_hpo={seed_hpo} (already present in terms.jsonl)")
                continue

            if args.debug and (si % max(1, int(args.debug_every)) == 0 or si == 1):
                dprint(f"[DBG] seed={seed_hpo} ({si}/{len(seeds)})")

            def _handle_term(r: Optional[TermRecord], kind: str) -> None:
                if not r:
                    if args.debug:
                        dprint(f"[DBG] term kind={kind} seed={seed_hpo} -> None")
                    return

                rr = r
                d = termrecord_to_dict(rr)
                terms_all.append(d)

                with open(terms_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    f.flush()

                if args.debug and (len(terms_all) == 1 or (len(terms_all) % max(1, int(args.debug_every)) == 0)):
                    phr = d.get("phrases")
                    phrases0_dbg = ""
                    if isinstance(phr, list):
                        phrases0_dbg = _short(phr[0]) if phr else ""
                    elif isinstance(phr, dict):
                        if phr:
                            try:
                                first_val = next(iter(phr.values()))
                            except Exception:
                                first_val = None
                            if first_val:
                                phrases0_dbg = _short(first_val)
                            else:
                                try:
                                    first_key = next(iter(phr.keys()))
                                except Exception:
                                    first_key = None
                                phrases0_dbg = _short(first_key) if first_key else ""
                    else:
                        phrases0_dbg = ""

                    dprint(
                        "[DBG][term] "
                        f"seed={seed_hpo} kind={d.get('kind')} "
                        f"used_hits={d.get('used_hits')} "
                        f"used_query={_short(d.get('used_query') or d.get('query'))} "
                        f"phrases0={phrases0_dbg}"
                    )

            for _ in range(max(0, int(args.n_seed0))):
                _handle_term(sampler.build_seed0(seed_hpo), "seed0")
            for _ in range(max(0, int(args.n_seed1))):
                _handle_term(sampler.build_seed1(seed_hpo), "seed1")
            for _ in range(max(0, int(args.n_seed2))):
                _handle_term(sampler.build_seed2(seed_hpo), "seed2")
            for _ in range(max(0, int(args.n_neg))):
                _handle_term(sampler.build_neg(seed_hpo), "neg")

        # build selected from ALL terms (old + new)
        terms_selected: List[Dict[str, Any]] = []
        for d in terms_all:
            if args.check_esearch:
                uh = d.get("used_hits")
                if isinstance(uh, int) and uh > 0:
                    terms_selected.append(d)
            else:
                terms_selected.append(d)

        write_jsonl(selected_path, terms_selected)
        print(f"[INFO] wrote terms_all={len(terms_all)} terms_selected={len(terms_selected)}")
        if args.debug:
            dprint(_head_file(terms_path, n_lines=args.head_lines, max_chars=args.head_chars))
            dprint(_head_file(selected_path, n_lines=args.head_lines, max_chars=args.head_chars))

    # -----------------------------
    # Part 2: Query -> PMIDs/PMCIDs (RESUME SAFE, with query-level LLM repair + PubMed abstract cache)
    # -----------------------------
    query_to_pmids_path = os.path.join(dir_pubmed, "query_to_pmids.jsonl")
    query_to_pmcids_path = os.path.join(dir_pubmed, "query_to_pmcids.jsonl")
    pmid_to_abstract_path = os.path.join(dir_pubmed, "pmid_to_abstract.jsonl")

    # NEW: HPO index + evidence_pool 路径
    hpo_to_pmids_path = os.path.join(dir_pubmed, "hpo_to_pmids.jsonl")
    hpo_to_pmcids_path = os.path.join(dir_pubmed, "hpo_to_pmcids.jsonl")
    evidence_pool_path = os.path.join(dir_pubmed, "evidence_pool.jsonl")  # 真正构建在后面

    # Build query_metas from selected terms
    query_metas: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in read_jsonl_iter(selected_path):
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

    query_to_pmids_existing = load_query_to_pmids_existing(query_to_pmids_path) if os.path.exists(query_to_pmids_path) else {}
    query_to_pmcids_existing = load_query_to_pmcids_existing(query_to_pmcids_path) if os.path.exists(query_to_pmcids_path) else {}

    # ---- Step 2.1 ESearch unique queries (skip done) ----
    query_to_pmids: Dict[str, List[str]] = dict(query_to_pmids_existing)

    todo_queries = [q for q in unique_queries if qkey(q, int(args.pmids_retmax)) not in done_qkeys]
    pbar_q1 = tqdm(todo_queries, desc="Stage3: ESearch todo queries", dynamic_ncols=True)

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

            meta0 = query_metas[q][0] if query_metas[q] else {"seed_hpo": None, "kind": None, "used_query": q}

            _append_and_flush_jsonl(query_to_pmids_path, {
                "seed_hpo": meta0.get("seed_hpo"),
                "kind": meta0.get("kind"),
                "used_query": q,
                "pmids_retmax": int(args.pmids_retmax),
                "pmids": pmids,
            })
            qcache.flush()

            if args.debug and (i == 1 or (i % max(1, int(args.debug_every)) == 0)):
                dprint(
                    f"[DBG][esearch] i={i}/{len(todo_queries)} "
                    f"q={_short(q, 160)} pmids={len(pmids)} pmid0={(pmids[0] if pmids else None)}"
                )
                dprint(_head_file(query_to_pmids_path, n_lines=args.head_lines, max_chars=args.head_chars))

        except Exception as e:
            _append_and_flush_jsonl(failures_path, {
                "stage": "pubmed_esearch",
                "used_query": q,
                "error": str(e),
            })

    # ---- Step 2.1b LLM repair only for true 0-hit queries ----
    bad_queries = [q for q in todo_queries if len(query_to_pmids.get(q, [])) == 0]
    n_repaired_queries = 0

    if bad_queries and args.use_llm_repair and llm is not None:
        pbar_fix = tqdm(bad_queries, desc="Stage3: LLM repair 0-hit queries", dynamic_ncols=True)
        for k, q in enumerate(pbar_fix, start=1):
            meta0 = query_metas[q][0] if query_metas[q] else {"seed_hpo": None, "kind": None, "used_query": q}
            seed_hpo = meta0.get("seed_hpo")
            kind = meta0.get("kind")

            try:
                prompt = (
                    "You are an expert biomedical librarian. "
                    "I have a PubMed search query that returned zero results. "
                    "Please rewrite it into a better PubMed search expression that is more likely to return "
                    "case reports or clinical studies related to the same phenotype.\n\n"
                    f"HPO_ID: {seed_hpo}\n"
                    f"Query kind: {kind}\n"
                    f"Original query:\n{q}\n\n"
                    "Output ONLY the new PubMed query string, without explanation."
                )
                repaired_q = None

                if hasattr(llm, "chat"):
                    repaired_q = llm.chat(prompt)  # type: ignore[attr-defined]
                elif hasattr(llm, "complete"):
                    repaired_q = llm.complete(prompt)  # type: ignore[attr-defined]
                else:
                    raise AttributeError("LLMClient has neither .chat nor .complete method")

                if not repaired_q:
                    continue

                repaired_q = str(repaired_q).strip().strip('"').strip("'")
                if not repaired_q:
                    continue

                pmids2 = safe_esearch_pmids(ncbi_client, repaired_q, retmax=int(args.pmids_retmax))
                pmids2 = [str(x) for x in pmids2 if str(x).strip()]

                query_to_pmids[q] = pmids2
                qcache.set_pmids(q, int(args.pmids_retmax), pmids2)
                qcache.flush()

                _append_and_flush_jsonl(query_to_pmids_path, {
                    "seed_hpo": seed_hpo,
                    "kind": kind,
                    "used_query": q,
                    "pmids_retmax": int(args.pmids_retmax),
                    "pmids": pmids2,
                    "repaired_query": repaired_q,
                })

                n_repaired_queries += 1

                if args.debug and (k == 1 or (k % max(1, int(args.debug_every)) == 0)):
                    dprint(
                        f"[DBG][repair] k={k}/{len(bad_queries)} "
                        f"orig_q={_short(q, 120)} "
                        f"repaired_q={_short(repaired_q, 120)} "
                        f"pmids={len(pmids2)}"
                    )
                    dprint(_head_file(query_to_pmids_path, n_lines=args.head_lines, max_chars=args.head_chars))

            except Exception as e:
                _append_and_flush_jsonl(failures_path, {
                    "stage": "pubmed_esearch_repair",
                    "used_query": q,
                    "error": str(e),
                })

    # ---- NEW: build pmid -> hpo_ids mapping for by-HPO indexing & evidence_pool ----
    pmid_to_hpos: DefaultDict[str, Set[str]] = defaultdict(set)
    pbar_pmid_hpo = tqdm(
        list(query_to_pmids.items()),
        desc="Stage3: build pmid_to_hpos",
        dynamic_ncols=True,
    )
    for q, pmids in pbar_pmid_hpo:
        metas = query_metas.get(q) or []
        seed_hpo = ""
        if metas:
            seed_hpo = (metas[0].get("seed_hpo") or "").strip()
        if not seed_hpo:
            continue
        for pmid in pmids:
            pmid_str = str(pmid).strip()
            if pmid_str:
                pmid_to_hpos[pmid_str].add(seed_hpo)

    # ---- Step 2.2: Build unique PMIDs ----
    unique_pmids_set: Set[str] = set()
    pbar_unique_pmids = tqdm(
        list(query_to_pmids.values()),
        desc="Stage3: build unique_pmids_set",
        dynamic_ncols=True,
    )
    for pmids in pbar_unique_pmids:
        for pmid in pmids:
            unique_pmids_set.add(str(pmid))

    unique_pmids: List[str] = sorted(unique_pmids_set)

    # ---- Step 2.2a: PubMed EFetch abstracts + weak evidence cache ----
    pmids_abstract_done: Set[str] = set()
    if os.path.exists(pmid_to_abstract_path):
        for obj in read_jsonl_iter(pmid_to_abstract_path):
            p = str(obj.get("pmid") or "").strip()
            if p:
                pmids_abstract_done.add(p)

    todo_pmids_for_abs = [p for p in unique_pmids if p not in pmids_abstract_done]

    pmid_to_xml: Dict[str, str] = {}
    missing_pubmed_pmids: List[str] = []

    for pmid in todo_pmids_for_abs:
        xml_hit = pubmed_xml_cache.get(pmid)
        if xml_hit:
            pmid_to_xml[pmid] = xml_hit
        else:
            missing_pubmed_pmids.append(pmid)

    if missing_pubmed_pmids:
        pbar_pubmed_fetch = tqdm(
            total=len(missing_pubmed_pmids),
            desc="Stage3: PubMed EFetch XML for abstracts (batch)",
            dynamic_ncols=True,
        )
        bs_pub = max(1, int(args.elink_batch_size))
        for chunk_start in range(0, len(missing_pubmed_pmids), bs_pub):
            chunk = missing_pubmed_pmids[chunk_start:chunk_start + bs_pub]
            try:
                rr = _rate_limited_call(ncbi_client.pubmed_efetch_xml_batch, chunk, batch_size=bs_pub)
                if not rr.get("ok"):
                    _append_and_flush_jsonl(failures_path, {
                        "stage": "pubmed_efetch_batch",
                        "error": rr.get("error", ""),
                        "text": rr.get("text", ""),
                        "failed_batch_size": len(chunk),
                    })
                else:
                    mp_xml = rr.get("pmid_to_xml") or {}
                    for pmid, xml_text in mp_xml.items():
                        pmid_str = str(pmid).strip()
                        if pmid_str and xml_text:
                            pmid_to_xml[pmid_str] = xml_text
                            pubmed_xml_cache.set(pmid_str, xml_text)
            except Exception as e:
                _append_and_flush_jsonl(failures_path, {
                    "stage": "pubmed_efetch_batch",
                    "error": str(e),
                    "failed_batch_size": len(chunk),
                })
            finally:
                pbar_pubmed_fetch.update(len(chunk))
        pbar_pubmed_fetch.close()

    if todo_pmids_for_abs:
        pbar_abs = tqdm(
            todo_pmids_for_abs,
            desc="Stage3: parse & cache PubMed abstracts (weak evidence)",
            dynamic_ncols=True,
        )
        for k, pmid in enumerate(pbar_abs, start=1):
            xml_text = pmid_to_xml.get(pmid, "")
            if not xml_text:
                _append_and_flush_jsonl(failures_path, {
                    "stage": "pubmed_parse_abstract",
                    "pmid": pmid,
                    "error": "missing xml_text",
                })
                continue
            try:
                parsed = ncbi_client.parse_pubmed_xml(xml_text)
            except Exception as e:
                _append_and_flush_jsonl(failures_path, {
                    "stage": "pubmed_parse_abstract",
                    "pmid": pmid,
                    "error": str(e),
                })
                continue

            if not isinstance(parsed, dict):
                parsed = {}

            title = (parsed.get("title") or "").strip()
            abstract = (parsed.get("abstract") or "").strip()
            journal = (parsed.get("journal") or parsed.get("source") or "").strip()
            year = parsed.get("year") or parsed.get("pub_year")

            entry = {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "has_abstract": bool(abstract),
            }
            _append_and_flush_jsonl(pmid_to_abstract_path, entry)

            if args.debug and (k == 1 or (k % max(1, int(args.debug_every)) == 0)):
                dprint(
                    f"[DBG][pubmed-abs] k={k}/{len(todo_pmids_for_abs)} pmid={pmid} "
                    f"has_abs={bool(abstract)} title={_short(title, 80)}"
                )

    # ---- NEW: build evidence_pool (global + per-HPO) ----
    evidence_pool_path = build_evidence_pool(
        pmid_to_abstract_path=pmid_to_abstract_path,
        pmid_to_hpos=pmid_to_hpos,
        pubmed_dir=pubmed_cache_dir,
    )

    # ---- Step 2.3 ELink PMIDs -> PMCIDs ----
    pmid_to_pmcids: Dict[str, List[str]] = {}
    if unique_pmids:
        pbar_elink = tqdm(total=len(unique_pmids), desc="Stage3: ELink unique PMIDs (batch)", dynamic_ncols=True)
        bs = max(1, int(args.elink_batch_size))
        for chunk_start in range(0, len(unique_pmids), bs):
            chunk = unique_pmids[chunk_start:chunk_start + bs]
            try:
                r = _rate_limited_call(ncbi_client.pubmed_elink_to_pmc_batch, chunk, batch_size=bs)
                if not r.get("ok"):
                    _append_and_flush_jsonl(failures_path, {
                        "stage": "pubmed_elink_batch",
                        "error": r.get("error", ""),
                        "text": r.get("text", ""),
                        "failed_batch_size": len(chunk),
                    })
                else:
                    mp2 = r.get("mapping") or {}
                    for pmid, pmc_list in mp2.items():
                        pmid_to_pmcids[str(pmid)] = [_normalize_pmcid(x) for x in (pmc_list or []) if x]
            except Exception as e:
                _append_and_flush_jsonl(failures_path, {
                    "stage": "pubmed_elink_batch",
                    "error": str(e),
                    "failed_batch_size": len(chunk),
                })
            finally:
                pbar_elink.update(len(chunk))
                if args.debug and (pbar_elink.n == len(unique_pmids) or (pbar_elink.n % max(1, int(args.debug_every)) == 0)):
                    dprint(f"[DBG][elink] progressed={pbar_elink.n}/{len(unique_pmids)} mapping_size={len(pmid_to_pmcids)}")
        pbar_elink.close()

    # ---- NEW: build pmcid -> hpo_ids mapping via pmid_to_hpos & pmid_to_pmcids ----
    pmcid_to_hpos: DefaultDict[str, Set[str]] = defaultdict(set)
    for pmid, hpos_set in pmid_to_hpos.items():
        pmc_list = pmid_to_pmcids.get(pmid) or []
        for pmcid in pmc_list:
            pmcid_norm = _normalize_pmcid(pmcid)
            if pmcid_norm and hpos_set:
                pmcid_to_hpos[pmcid_norm].update(hpos_set)

    # ---- Step 2.4 Build PMCIDs per query (skip already done) ----
    query_to_pmcids: Dict[str, List[str]] = dict(query_to_pmcids_existing)

    pmcid_index_path = os.path.join(dir_metrics, "pmcid_index.jsonl")
    pmcid_meta_seen: Set[str] = set()
    if os.path.exists(pmcid_index_path):
        for obj in read_jsonl_iter(pmcid_index_path):
            pmcid = _normalize_pmcid(obj.get("pmcid") or "")
            if pmcid:
                pmcid_meta_seen.add(pmcid)

    # NEW: per-HPO PMC index dir（依旧保留）
    pmcid_by_hpo_dir = os.path.join(dir_phaseA, "by_hpo")
    ensure_dir(pmcid_by_hpo_dir)

    all_pmcids: List[str] = []
    seen_pmcids: Set[str] = set()
    for pmcids in query_to_pmcids.values():
        for pmcid in pmcids:
            pmcid = _normalize_pmcid(pmcid)
            if pmcid and pmcid not in seen_pmcids:
                seen_pmcids.add(pmcid)
                all_pmcids.append(pmcid)

    pbar_q2 = tqdm(unique_queries, desc="Stage3: build PMCIDs per query (resume-safe)", dynamic_ncols=True)

    for j, q in enumerate(pbar_q2, start=1):
        k = qkey(q, int(args.pmids_retmax))
        if k in done_qkeys:
            continue

        pmids = query_to_pmids.get(q, [])
        pmcids = _join_pmcids_from_pmid_map(pmids, pmid_to_pmcids)

        if args.pmc_per_query and int(args.pmc_per_query) > 0:
            pmcids = pmcids[: int(args.pmc_per_query)]

        qcache.set_pmcids(q, int(args.pmids_retmax), pmcids)
        qcache.flush()

        meta0 = query_metas[q][0] if query_metas[q] else {"seed_hpo": None, "kind": None, "used_query": q}

        _append_and_flush_jsonl(query_to_pmcids_path, {
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
                _append_and_flush_jsonl(pmcid_index_path, {
                    "pmcid": pmcid,
                    "from_seed_hpo": meta0.get("seed_hpo"),
                    "kind": meta0.get("kind"),
                    "used_query": q,
                })

            # per-HPO PMC index（保持原样，同时用 pmcid_to_hpos 兜底）
            hpo_ids_for_pmc = sorted(pmcid_to_hpos.get(pmcid, []))
            if not hpo_ids_for_pmc and meta0.get("seed_hpo"):
                hpo_ids_for_pmc = [meta0.get("seed_hpo")]
            for hpo_id in hpo_ids_for_pmc:
                if not hpo_id:
                    continue
                hpo_pmc_path = os.path.join(pmcid_by_hpo_dir, f"{hpo_id}.jsonl")
                _append_and_flush_jsonl(hpo_pmc_path, {
                    "hpo_id": hpo_id,
                    "pmcid": pmcid,
                    "from_seed_hpo": meta0.get("seed_hpo"),
                    "kind": meta0.get("kind"),
                    "used_query": q,
                })

        if args.debug and (j == 1 or (j % max(1, int(args.debug_every)) == 0)):
            dprint(f"[DBG][q->pmc] j={j}/{len(unique_queries)} pmcids={len(pmcids)} pmcid0={(pmcids[0] if pmcids else None)} q={_short(q, 140)}")
            dprint(_head_file(query_to_pmcids_path, n_lines=args.head_lines, max_chars=args.head_chars))

    # ---- NEW: build HPO-level indexes (hpo_to_pmids / hpo_to_pmcids) ----
    n_hpo_pmids, n_hpo_pmcids = build_and_write_hpo_indexes(
        pmid_to_hpos=pmid_to_hpos,
        pmcid_to_hpos=pmcid_to_hpos,
        hpo_to_pmids_path=hpo_to_pmids_path,
        hpo_to_pmcids_path=hpo_to_pmcids_path,
    )
    print(f"[INFO] hpo_to_pmids rows={n_hpo_pmids} hpo_to_pmcids rows={n_hpo_pmcids}")

    # -----------------------------
    # Part 3: PMC EFetch + PhaseA (RESUME SAFE)
    # -----------------------------
    merged_case = os.path.join(dir_phaseA_merged, "case_chunks.jsonl")
    merged_mentions = os.path.join(dir_phaseA_merged, "mentions_candidates.jsonl")
    ensure_dir(dir_phaseA_merged)

    if not os.path.exists(merged_case):
        open(merged_case, "w", encoding="utf-8").close()
    if not os.path.exists(merged_mentions):
        open(merged_mentions, "w", encoding="utf-8").close()

    done_pmcids = load_done_pmcids_from_done(dir_phaseA)
    todo_pmcids = [p for p in all_pmcids if _normalize_pmcid(p) not in done_pmcids]
    todo_pmcids = [_normalize_pmcid(x) for x in todo_pmcids if _normalize_pmcid(x)]

    print(f"[RESUME] phaseA done_pmcids={len(done_pmcids)}  todo_pmcids={len(todo_pmcids)}  all_pmcids={len(all_pmcids)}")
    if args.debug:
        dprint(f"[DBG] merged_case_lines={_count_jsonl_lines(merged_case)} merged_mentions_lines={_count_jsonl_lines(merged_mentions)}")
        dprint(_head_file(merged_case, n_lines=args.head_lines, max_chars=args.head_chars))
        dprint(_head_file(merged_mentions, n_lines=args.head_lines, max_chars=args.head_chars))

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
                    _append_and_flush_jsonl(failures_path, {
                        "stage": "pmc_efetch_batch",
                        "error": rr.get("error", ""),
                        "text": rr.get("text", ""),
                        "failed_batch_size": len(chunk),
                    })
                else:
                    mp3 = rr.get("pmcid_to_xml") or {}
                    for pmcid, xml_text in mp3.items():
                        pmcid = _normalize_pmcid(pmcid)
                        if pmcid and xml_text:
                            pmcid_to_xml[pmcid] = xml_text
                            pmc_xml_cache.set(pmcid, xml_text)
            except Exception as e:
                _append_and_flush_jsonl(failures_path, {
                    "stage": "pmc_efetch_batch",
                    "error": str(e),
                    "failed_batch_size": len(chunk),
                })
            finally:
                pbar_fetch.update(len(chunk))
                if args.debug and (pbar_fetch.n == len(missing_pmcids) or (pbar_fetch.n % max(1, int(args.debug_every)) == 0)):
                    dprint(f"[DBG][efetch] progressed={pbar_fetch.n}/{len(missing_pmcids)} xml_cached_now={len(pmcid_to_xml)}")
        pbar_fetch.close()

    lock_merge = threading.Lock()
    ok, fail, skipped = 0, 0, 0
    total_chunks, total_cands = 0, 0

    phaseA_status_path = os.path.join(dir_metrics, "phaseA_status.jsonl")

    pbar_extract = tqdm(total=len(todo_pmcids), desc="Stage3: PhaseA parse/extract PMCIDs (resume-safe)", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=int(args.threads)) as ex:
        futs = {}
        for pmcid in todo_pmcids:
            xml_text = pmcid_to_xml.get(pmcid, "")
            futs[ex.submit(run_one_pmcid_extract_from_xml, pmcid, xml_text, cfg, dir_phaseA)] = pmcid

        for idx, fut in enumerate(as_completed(futs), start=1):
            pmcid = futs[fut]
            try:
                pmcid, n_chunks, n_cands, was_skipped, head_chunk, head_cand = fut.result()
                if was_skipped:
                    skipped += 1
                    _append_and_flush_jsonl(phaseA_status_path, {
                        "pmcid": pmcid,
                        "skipped": True,
                        "n_chunks": 0,
                        "n_candidates": 0,
                        "head_chunk": "",
                        "head_candidate_surface": "",
                        "ts": time.time(),
                    })
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
                        f.flush()
                    with open(merged_mentions, "a", encoding="utf-8") as f:
                        for ln in mention_lines:
                            f.write(ln + "\n")
                        f.flush()

                ok += 1

                _append_and_flush_jsonl(phaseA_status_path, {
                    "pmcid": pmcid,
                    "skipped": False,
                    "n_chunks": int(n_chunks),
                    "n_candidates": int(n_cands),
                    "head_chunk": head_chunk,
                    "head_candidate_surface": head_cand,
                    "ts": time.time(),
                })

                if args.debug and (idx == 1 or (idx % max(1, int(args.debug_every)) == 0)):
                    dprint(f"[DBG][phaseA] idx={idx}/{len(todo_pmcids)} pmcid={pmcid} chunks={n_chunks} cands={n_cands}")
                    dprint(f"  head_chunk={_short(head_chunk, 200)}")
                    dprint(f"  head_cand ={_short(head_cand, 120)}")
                    dprint(_head_file(merged_case, n_lines=args.head_lines, max_chars=args.head_chars))
                    dprint(_head_file(merged_mentions, n_lines=args.head_lines, max_chars=args.head_chars))

            except Exception as e:
                fail += 1
                _append_and_flush_jsonl(failures_path, {
                    "stage": "phaseA_extract",
                    "pmcid": pmcid,
                    "error": str(e),
                })
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
            "n_bad_queries": len(bad_queries),
            "n_repaired_queries": int(n_repaired_queries),
            "cache_dir": pubmed_cache_dir,
            "hpo_to_pmids_path": hpo_to_pmids_path,
            "hpo_to_pmcids_path": hpo_to_pmcids_path,
            "n_hpo_pmids_rows": n_hpo_pmids,
            "n_hpo_pmcids_rows": n_hpo_pmcids,
            "evidence_pool_path": evidence_pool_path,
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
            "phaseA_status_jsonl": os.path.join(dir_metrics, "phaseA_status.jsonl"),
        },

        "paths": {
            "terms_all": os.path.join(dir_terms, "terms.jsonl"),
            "terms_selected": os.path.join(dir_terms, "terms_selected.jsonl"),
            "query_to_pmids": query_to_pmids_path,
            "query_to_pmcids": query_to_pmcids_path,
            "pmid_to_abstract": pmid_to_abstract_path,
            "pmcid_index": pmcid_index_path,
            "hpo_to_pmids": hpo_to_pmids_path,
            "hpo_to_pmcids": hpo_to_pmcids_path,
            "evidence_pool": evidence_pool_path,
            "failures": failures_path,
            "summary": os.path.join(dir_metrics, "summary.json"),
            "run_config": os.path.join(run_dir, "run_config.json"),
            "manifest": os.path.join(run_dir, "manifest.json"),
        },
    }

    # 写 summary.json
    _atomic_write_json(os.path.join(dir_metrics, "summary.json"), summary)

    # NEW: manifest.json（可作为 PhaseB / 评估 / 数据资产的入口清单）
    manifest = {
        "run_id": run_id,
        "run_dir": run_dir,
        "created_at": now_stamp(),
        "core_outputs": summary["paths"],
        "stats": {
            "n_seeds": len(seeds),
            "n_unique_pmids": len(unique_pmids),
            "n_unique_pmcids": len(all_pmcids),
            "n_hpo_pmids_rows": n_hpo_pmids,
            "n_hpo_pmcids_rows": n_hpo_pmcids,
            "phaseA_ok_pmcids": ok,
            "phaseA_fail_pmcids": fail,
        },
    }
    _atomic_write_json(os.path.join(run_dir, "manifest.json"), manifest)

    print("\n[DONE] Full Stage3 run finished (resume-safe).")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[HINT] PhaseB 在写 corpus.jsonl 时，建议对 normalized_text 使用 _sha1() 存一个 'sha1_text_norm' 字段。")


if __name__ == "__main__":
    main()

"""
Example:

python run_stage3_phaseA_full.py \
  --queries_jsonl "$QUERIES_JSONL" \
  --neighbors_jsonl "$NEIGHBORS_JSONL" \
  --email "2018912302@qq.com" \
  --out_dir "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/out_phaseA" \
  --threads 3 \
  --polite_sleep 0.34 \
  --seed 42 \
  --n_seed0 1 --n_seed1 1 --n_seed2 1 --n_neg 1 \
  --max_seeds 0 \
  --pmc_per_query 8 \
  --llm_model deepseek-chat \
  --llm_base_url https://api.deepseek.com \
  --debug --debug_every 20 --head_lines 2 --head_chars 220
"""
