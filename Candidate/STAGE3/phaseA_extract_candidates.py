#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
phaseA_extract_candidates.py  (FIXED)

Phase A:
- Fetch PMC fulltext via pubmed_pmc_client.py (PubMedPMCClient)
- Build case-like chunks
- Extract RULE-BASED mention candidates (measurements / temporal)
- Output:
  - case_chunks.jsonl
  - mentions_candidates.jsonl

Fixes vs your current revision:
1) Sentence splitter robust to "minute.Blood" (no whitespace after punctuation in NXML)
2) Clause-first context for measurement-like mentions (BP/HR/ECG/LAB): avoid cross-clause noise
3) Polarity flags (LOW/HIGH/NORMAL) only for LAB/physio measurements, NOT for temporal/frequency
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Local client (same dir)
from pubmed_pmc_client import PubMedPMCClient, NCBIConfig, _normalize_pmcid  # type: ignore


# -----------------------------
# IO helpers
# -----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def compact_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# -----------------------------
# Sentence / clause context
# -----------------------------
# FIX #1: punctuation boundary does NOT require whitespace
# - handles "...minute.Blood..." in XML
_SENT_SPLIT = re.compile(r"([.!?;。！？；](?:\s+|$)|\n+)")

# Clause boundaries inside a sentence (for cleaner measurement context)
_CLAUSE_SPLIT = re.compile(r"(,\s+|;\s+|:\s+)")

def iter_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return list of (start,end) sentence-like spans.
    Robust to punctuation without whitespace.
    """
    spans: List[Tuple[int, int]] = []
    if not text:
        return spans

    parts = _SENT_SPLIT.split(text)
    cur = ""
    cur_start = 0
    cur_pos = 0

    # normalize parts list (keep separators too)
    tmp: List[str] = [p for p in parts if p]

    for p in tmp:
        if _SENT_SPLIT.match(p):
            # close sentence including separator
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

    # merge ultra-short into previous
    merged: List[Tuple[int, int]] = []
    for st, ed in spans:
        if not merged:
            merged.append((st, ed))
            continue
        if ed - st < 25:
            pst, ped = merged[-1]
            merged[-1] = (pst, ed)
        else:
            merged.append((st, ed))
    return merged


def _pick_containing_span(spans: List[Tuple[int, int]], a: int, b: int) -> Tuple[int, int]:
    # pick containing; else nearest center
    for st, ed in spans:
        if st <= a and b <= ed:
            return st, ed
    mid = (a + b) / 2.0
    return min(spans, key=lambda x: abs(((x[0] + x[1]) / 2.0) - mid))


def _clause_first_within_sentence(sentence: str, rel_span: Tuple[int, int], side_extra: int = 40) -> str:
    """
    Given a sentence string and a span relative to this sentence,
    return the clause containing the span (split by comma/;/:) with slight extension.
    """
    if not sentence:
        return ""

    a, b = rel_span
    a = max(0, a)
    b = min(len(sentence), b)

    # build clause spans
    parts = _CLAUSE_SPLIT.split(sentence)
    # Rebuild with separators so indices align
    clauses: List[Tuple[int, int]] = []
    buf = ""
    buf_start = 0
    cur_pos = 0

    tmp: List[str] = [p for p in parts if p]
    for p in tmp:
        if _CLAUSE_SPLIT.match(p):
            # treat separator as end of a clause
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

    # allow a tiny extension inside sentence, but DO NOT cross sentence boundary
    st2 = max(0, cst - side_extra)
    ed2 = min(len(sentence), ced + side_extra)
    return compact_ws(sentence[st2:ed2])


def sentence_first_context(paragraph: str, match_span: Tuple[int, int],
                           side_extra: int = 60,
                           clause_first: bool = False) -> str:
    """
    Find sentence containing match_span.
    If clause_first=True: return the clause containing the mention within that sentence.
    """
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

    # sentence only (with small extension within paragraph, but do not cross next sentence boundary)
    # We'll just return the sentence itself; side_extra here is intentionally small/no-cross.
    # (If you want, you can extend to include adjacent clauses, but avoid crossing into next sentence.)
    return compact_ws(sent)


# -----------------------------
# Polarity hints (FIX #3: gated by measurement type)
# -----------------------------
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


# -----------------------------
# Candidate extraction rules
# -----------------------------

# Blood pressure: 94/60 mmHg
_RE_BP = re.compile(r"\b(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})\s*(?P<unit>mmhg)\b", re.IGNORECASE)

# HR (optional explicit): 112 beats/minute OR 112 bpm
_RE_HR = re.compile(r"\b(?P<val>\d{2,3})\s*(?P<unit>bpm|beats\/min|beats\/minute)\b", re.IGNORECASE)

# Generic value+unit including ranges:
# - 861 ms
# - 0.47 mmol/L
# - 30–45 seconds
_RE_VAL_UNIT = re.compile(
    r"(?P<val>(?:\d+(?:\.\d+)?)|(?:\d+\s*[–-]\s*\d+(?:\.\d+)?))\s*"
    r"(?P<unit>mmol\/l|mg\/dl|ms|s|sec|seconds|minutes|min|hours|h|days|day|weeks|week|months|month|years|year|%|mmhg)\b",
    re.IGNORECASE
)

# Reference range in same context
_RE_RANGE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*[–-]\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<unit>mmol\/l|mg\/dl|ms|%|mmhg)\b",
    re.IGNORECASE
)

# High-signal anchors (no generic unit spam)
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

    # Add a few literal triggers for audit (kept small)
    lit_keep = [
        "over the past", "for the previous", "every", "per day", "per month", "lasting",
        "blood pressure", "heart rate", "ejection fraction", "bazett", "electrocardiogram", "ionized calcium",
    ]
    lctx = ctx.lower()
    for lit in lit_keep:
        if lit in lctx:
            hits.append(lit)

    # de-dup
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
    # things where low/high/normal flags are meaningful in clinical sense
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

    # FIX #3: polarity flags ONLY for lab/physio units (NOT for temporal)
    if _is_lab_or_physio_unit(unit_l) and not _is_temporal_unit(unit_l):
        opts += polarity_flags(ctx)

    # de-dup preserve order
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
        return ul  # keep canonical lower form
    if ul == "bpm":
        return "bpm"
    if ul == "mmol/l":
        return "mmol/L"
    if ul == "mg/dl":
        return "mg/dL"
    return u


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Chunk:
    pmcid: str
    section: str
    para_idx: int
    text: str

    @property
    def chunk_key(self) -> str:
        return f"{self.pmcid}|{self.section}|{self.para_idx}"


# -----------------------------
# Phase A Extractor class
# -----------------------------

class PhaseAExtractor:
    def __init__(self, client: PubMedPMCClient):
        self.client = client

    def fetch_case_chunks_by_term(self, term: str, retmax: int = 20, need_pmc: bool = False) -> Tuple[str, List[Chunk]]:
        sr = self.client.pubmed_esearch(term, retmax=retmax)
        if not sr.get("ok"):
            raise RuntimeError(f"PubMed ESearch failed: {sr}")
        pmids = sr.get("pmids") or []
        if not pmids:
            raise RuntimeError("No PubMed hits for term.")

        lr = self.client.pubmed_elink_to_pmc(pmids)
        if not lr.get("ok"):
            raise RuntimeError(f"PubMed ELink->PMC failed: {lr}")
        pmcids = lr.get("pmcids") or []

        if need_pmc and not pmcids:
            raise RuntimeError("need_pmc=True but no PMCIDs found.")
        if not pmcids:
            raise RuntimeError("No PMCIDs found; cannot fetch full text.")

        pmcid0 = pmcids[0]
        return pmcid0, self.fetch_case_chunks_by_pmcid(pmcid0)

    def fetch_case_chunks_by_pmcid(self, pmcid: str) -> List[Chunk]:
        pmcid = _normalize_pmcid(pmcid)
        fr = self.client.pmc_efetch_xml(pmcid)
        if not fr.get("ok"):
            raise RuntimeError(f"PMC EFetch failed: {fr}")
        parsed = self.client.parse_pmc_xml(fr["xml"])
        if not parsed.get("ok"):
            raise RuntimeError(f"PMC XML parse failed: {parsed}")

        sections = parsed.get("case_sections") or []
        if not sections:
            sections = parsed.get("sections") or []

        chunks: List[Chunk] = []
        for sec in sections:
            sec_title = compact_ws(sec.get("title") or "UNKNOWN")
            paras = sec.get("paras") or []
            for i, p in enumerate(paras):
                t = compact_ws(p)
                if t:
                    chunks.append(Chunk(pmcid=pmcid, section=sec_title, para_idx=i, text=t))

        # add abstract (optional)
        abs_paras = parsed.get("abstract_paras") or []
        for i, p in enumerate(abs_paras[:3]):
            t = compact_ws(p)
            if t:
                chunks.insert(0, Chunk(pmcid=pmcid, section="Abstract", para_idx=i, text=t))

        return chunks

    def extract_candidates_from_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for ch in chunks:
            para = ch.text
            bp_spans: List[Tuple[int, int]] = []

            # --- BP first (specific)
            for m in _RE_BP.finditer(para):
                st, ed = m.span()
                bp_spans.append((st, ed))
                ctx = sentence_first_context(para, (st, ed), clause_first=True)  # FIX #2: clause-first
                anchors = find_anchors(ctx)
                cand = {
                    "type": "measurement",
                    "chunk_key": ch.chunk_key,
                    "pmcid": ch.pmcid,
                    "section": ch.section,
                    "para_idx": ch.para_idx,
                    "span_char": [st, ed],
                    "surface": para[st:ed],
                    "unit": "mmHg",
                    "value": {"systolic": float(m.group("sys")), "diastolic": float(m.group("dia"))},
                    "anchors_hit": anchors,
                    "label_options": label_options_from_unit_and_anchors("mmHg", anchors, ctx),
                    "context": ctx,
                }
                out.append(cand)

            # --- HR explicit (optional)
            for m in _RE_HR.finditer(para):
                st, ed = m.span()
                # skip if within BP span
                if any((st >= bst and ed <= bed) for bst, bed in bp_spans):
                    continue
                ctx = sentence_first_context(para, (st, ed), clause_first=True)
                anchors = find_anchors(ctx)
                unit = (m.group("unit") or "").strip()
                cand = {
                    "type": "measurement",
                    "chunk_key": ch.chunk_key,
                    "pmcid": ch.pmcid,
                    "section": ch.section,
                    "para_idx": ch.para_idx,
                    "span_char": [st, ed],
                    "surface": para[st:ed],
                    "unit": normalize_unit(unit),
                    "value": float(m.group("val")),
                    "anchors_hit": anchors,
                    "label_options": label_options_from_unit_and_anchors(unit, anchors, ctx),
                    "context": ctx,
                }
                out.append(cand)

            # --- generic val+unit
            for m in _RE_VAL_UNIT.finditer(para):
                st, ed = m.span()

                # Skip if overlaps with BP span
                if any(not (ed <= bst or st >= bed) for bst, bed in bp_spans):
                    continue

                surface = para[st:ed]
                unit = (m.group("unit") or "").strip()
                val_raw = (m.group("val") or "").strip()

                # parse value
                value: Any
                if re.search(r"[–-]", val_raw):
                    mm = re.split(r"\s*[–-]\s*", val_raw)
                    try:
                        value = {"low": float(mm[0]), "high": float(mm[1])}
                    except Exception:
                        continue
                else:
                    try:
                        value = float(val_raw)
                    except Exception:
                        continue

                # FIX #2: clause-first context for physio/lab/ecg (ms/mmHg/%/mmol/L/mg/dL)
                unit_l = unit.lower()
                clause_first = _is_lab_or_physio_unit(unit_l) and not _is_temporal_unit(unit_l)
                ctx = sentence_first_context(para, (st, ed), clause_first=clause_first)

                anchors = find_anchors(ctx)

                # ref range in ctx (same unit)
                ref_range = None
                for rm in _RE_RANGE.finditer(ctx):
                    if (rm.group("unit") or "").lower() == unit_l:
                        try:
                            ref_range = {"low": float(rm.group("lo")), "high": float(rm.group("hi")), "unit": normalize_unit(unit)}
                            break
                        except Exception:
                            pass

                unit_norm = normalize_unit(unit)
                opts = label_options_from_unit_and_anchors(unit, anchors, ctx)

                cand: Dict[str, Any] = {
                    "type": "measurement",
                    "chunk_key": ch.chunk_key,
                    "pmcid": ch.pmcid,
                    "section": ch.section,
                    "para_idx": ch.para_idx,
                    "span_char": [st, ed],
                    "surface": surface,
                    "unit": unit_norm,
                    "value": value,
                    "anchors_hit": anchors,
                    "label_options": opts,
                    "context": ctx,
                }
                if ref_range:
                    cand["ref_range"] = ref_range
                out.append(cand)

        return out


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", type=str, required=True, help="NCBI polite use email")
    ap.add_argument("--api-key", type=str, default=None, help="Optional NCBI API key")
    ap.add_argument("--tool", type=str, default="hpo-agent-phaseA", help="NCBI tool name")
    ap.add_argument("--polite-sleep", type=float, default=0.34)

    ap.add_argument("--term", type=str, default="", help="PubMed search term")
    ap.add_argument("--retmax", type=int, default=20)
    ap.add_argument("--need-pmc", action="store_true")

    ap.add_argument("--pmcid", type=str, default="", help="Direct PMCID fetch (skip PubMed)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output dir")
    ap.add_argument("--max-chunks", type=int, default=0, help="If >0, truncate chunks for testing")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    cfg = NCBIConfig(
        email=args.email,
        tool=args.tool,
        api_key=args.api_key,
        polite_sleep=float(args.polite_sleep),
    )
    client = PubMedPMCClient(cfg)
    extractor = PhaseAExtractor(client)

    # Fetch chunks
    if args.pmcid.strip():
        pmcid_used = _normalize_pmcid(args.pmcid.strip())
        print(f"[PhaseA] Direct fetch PMCID={pmcid_used}")
        chunks = extractor.fetch_case_chunks_by_pmcid(pmcid_used)
    else:
        term = args.term.strip()
        if not term:
            raise ValueError("Provide --term or --pmcid.")
        print(f"[PhaseA] PubMed search term={term!r} retmax={args.retmax}")
        pmcid_used, chunks = extractor.fetch_case_chunks_by_term(term, retmax=args.retmax, need_pmc=bool(args.need_pmc))
        print(f"[PhaseA] Using PMCID={pmcid_used}")

    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: int(args.max_chunks)]

    # Save case_chunks.jsonl
    case_chunks_path = os.path.join(args.out_dir, "case_chunks.jsonl")
    case_rows: List[Dict[str, Any]] = []
    for ch in chunks:
        case_rows.append({
            "chunk_key": ch.chunk_key,
            "pmcid": ch.pmcid,
            "section": ch.section,
            "para_idx": ch.para_idx,
            "text": ch.text,
        })
    write_jsonl(case_chunks_path, case_rows)

    # Extract candidates
    cands = extractor.extract_candidates_from_chunks(chunks)
    mentions_path = os.path.join(args.out_dir, "mentions_candidates.jsonl")
    write_jsonl(mentions_path, cands)

    print("\n[DONE] Output:")
    print(f" - {case_chunks_path} (chunks={len(case_rows)})")
    print(f" - {mentions_path} (candidates={len(cands)})")

    print("\n[Sample candidates]")
    for x in cands[:6]:
        print(json.dumps(x, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
