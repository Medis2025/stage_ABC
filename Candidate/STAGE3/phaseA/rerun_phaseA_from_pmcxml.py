#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rerun_phaseA_from_pmcxml.py
===========================

Re-run PhaseA mentions extraction directly from cached PMC XML (NO reuse of old jsonl).

Input:
  - <run_dir>/.cache/pmc_xml/*.xml   (cached by your Stage3 PhaseA run)

Output (default):
  - <run_dir>/phaseA_rerun/merged/case_chunks.jsonl
  - <run_dir>/phaseA_rerun/merged/mentions_candidates.jsonl
  - <run_dir>/phaseA_rerun/failures.jsonl

Key points:
  - DO NOT touch original PhaseA outputs.
  - NO client, NO requests, NO API.
  - Parse PMC XML with a local parse_pmc_xml() implemented in this file.
  - Extract measurement spans using regex.
  - FIX anchors_hit bug: anchors_hit is phrase/label based (not per-character).
  - Multi-threaded, resume-safe per PMCID (DONE markers).
  - Debug logs for each major step (optional, controlled by --debug / --debug_every).
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import hashlib
from typing import Any, Dict, List, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# =============================================================================
# IO utils
# =============================================================================

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def append_lines(path: str, lines: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as fo:
        for ln in lines:
            fo.write(ln + "\n")
        fo.flush()

def compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def now_ts() -> float:
    return time.time()

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def log_debug(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)

# =============================================================================
# PMC helpers
# =============================================================================

def _normalize_pmcid(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return x
    if x.upper().startswith("PMC"):
        return "PMC" + re.sub(r"^PMC", "", x, flags=re.IGNORECASE)
    if x.isdigit():
        return "PMC" + x
    return x

def load_pmcids_from_cache(xml_dir: str) -> List[str]:
    pmcids: List[str] = []
    if not os.path.isdir(xml_dir):
        return pmcids
    for fn in os.listdir(xml_dir):
        if not fn.lower().endswith(".xml"):
            continue
        pmcid = _normalize_pmcid(fn[:-4])
        if pmcid:
            pmcids.append(pmcid)
    return sorted(list(dict.fromkeys(pmcids)))

def read_xml(xml_dir: str, pmcid: str) -> Optional[str]:
    pmcid = _normalize_pmcid(pmcid)
    p = os.path.join(xml_dir, f"{pmcid}.xml")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

# =============================================================================
# sentence-first context (same spirit as your PhaseA)
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
    tmp = [p for p in parts if p]
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
        spans.append((cur_start, cur_start + len(cur)))

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

def sentence_first_context(paragraph: str, match_span: Tuple[int, int], clause_first: bool = False) -> str:
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

    if not clause_first:
        return compact_ws(sent)

    rel_a, rel_b = a - sst, b - sst
    parts = _CLAUSE_SPLIT.split(sent)
    clauses: List[Tuple[int, int]] = []
    buf = ""
    buf_start = 0
    cur_pos = 0
    tmp = [p for p in parts if p]
    for p in tmp:
        if _CLAUSE_SPLIT.match(p):
            buf += p
            cur_pos += len(p)
            clauses.append((buf_start, buf_start + len(buf)))
            buf = ""
            buf_start = cur_pos
        else:
            buf += p
            cur_pos += len(p)
    if buf.strip():
        clauses.append((buf_start, buf_start + len(buf)))

    if not clauses:
        return compact_ws(sent)

    cst, ced = _pick_containing_span(clauses, rel_a, rel_b)
    st2 = max(0, cst - 40)
    ed2 = min(len(sent), ced + 40)
    return compact_ws(sent[st2:ed2])

# =============================================================================
# anchors / label options (FIXED, no per-character)
# =============================================================================

_ANCHOR_SETS = {
    "ECG": [r"\bqtc\b", r"\bqt\b", r"\bbazett\b", r"\belectrocardiogram\b", r"\becg\b"],
    "BP":  [r"\bblood pressure\b", r"\bmmhg\b", r"\bhypertension\b", r"\bhypotension\b"],
    "HR":  [r"\bheart rate\b", r"\bbeats\/min\b", r"\bbeats\/minute\b", r"\bbpm\b"],
    "CALCIUM": [r"\bionized calcium\b", r"\bserum calcium\b", r"\bca2\+\b", r"\bcalcium\b"],
    "MAGNESIUM": [r"\bmagnesium\b", r"\bmg2\+\b"],
    "EF":  [r"\bejection fraction\b", r"\bef\b", r"\blvef\b"],
    "FREQ": [r"\bevery\b", r"\bper day\b", r"\bdaily\b", r"\btimes\b", r"\bepisodes\b"],
    "DUR":  [r"\bover the past\b", r"\bfor the previous\b", r"\blast(?:ing|ed)\b", r"\bduration\b", r"\bfollow-?up\b"],
}

_LITERAL_KEEP = [
    "over the past", "for the previous", "every", "per day", "per month",
    "follow-up", "follow up", "blood pressure", "heart rate",
    "ejection fraction", "bazett", "electrocardiogram", "ionized calcium",
    "ogtt",
]

def find_anchors_fixed(ctx: str) -> List[str]:
    if not ctx:
        return []
    hits: List[str] = []
    lctx = ctx.lower()

    for label, patterns in _ANCHOR_SETS.items():
        for pat in patterns:
            if re.search(pat, ctx, flags=re.IGNORECASE):
                hits.append(label)
                break

    for lit in _LITERAL_KEEP:
        if lit in lctx:
            hits.append(lit)

    out: List[str] = []
    seen = set()
    for h in hits:
        h2 = (h or "").strip()
        if h2 and h2 not in seen:
            seen.add(h2)
            out.append(h2)
    return out

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

def _is_temporal_unit(unit_l: str) -> bool:
    return unit_l in ("days", "day", "weeks", "week", "months", "month", "years", "year",
                      "hours", "h", "minutes", "minute", "min", "seconds", "sec", "s")

def _is_lab_or_physio_unit(unit_l: str) -> bool:
    return unit_l in ("mmol/l", "mg/dl", "ms", "mmhg", "%", "bpm", "beats/min", "beats/minute")

def label_options_from_unit(unit: str, anchors: List[str], ctx: str) -> List[str]:
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
        x2 = str(x).strip()
        if x2 and x2 not in seen:
            seen.add(x2)
            out.append(x2)
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

# =============================================================================
# regex for measurement extraction
# =============================================================================

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

# =============================================================================
# Local PMC XML parser (NO client)
# =============================================================================

def _compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _safe_text(el) -> str:
    if el is None:
        return ""
    parts: List[str] = []
    for t in el.itertext():
        if t:
            parts.append(t)
    return _compact_ws(" ".join(parts))

def _lower(s: str) -> str:
    return (s or "").strip().lower()

def _is_caseish_title(title: str) -> bool:
    t = _lower(title)
    if not t:
        return False
    keys = [
        "case", "case report", "case presentation", "case description",
        "clinical", "clinical features", "clinical presentation",
        "patient", "patients", "proband",
        "history", "physical examination",
        "findings", "phenotype",
    ]
    return any(k in t for k in keys)

def parse_pmc_xml(xml_text: str, debug: bool = False, pmcid: str = "") -> Dict[str, Any]:
    """
    Best-effort parsing of PMC XML (NXML-like). Extract:
      - title
      - abstract paragraphs
      - sections -> paragraphs
      - case_sections subset (heuristic on section title)
    """
    xml_text = xml_text or ""
    if not xml_text.strip():
        return {"ok": False, "error": "empty xml_text"}

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
    except Exception as e:
        return {"ok": False, "error": f"XML parse error: {e}"}

    title_el = root.find(".//article-title")
    title = _safe_text(title_el)

    abs_paras: List[str] = []
    for p in root.findall(".//abstract//p"):
        t = _safe_text(p)
        if t:
            abs_paras.append(t)

    sections: List[Dict[str, Any]] = []
    case_sections: List[Dict[str, Any]] = []

    # parse body sections
    for sec in root.findall(".//body//sec"):
        sec_title = _safe_text(sec.find("./title"))
        paras: List[str] = []
        for p in sec.findall(".//p"):
            t = _safe_text(p)
            if t:
                paras.append(t)

        if not paras and not sec_title:
            continue

        item = {"title": sec_title, "n_paras": len(paras), "paras": paras}
        sections.append(item)
        if _is_caseish_title(sec_title):
            case_sections.append(item)

    if debug:
        log_debug(True, f"[DBG][parse_pmc_xml]{'['+pmcid+']' if pmcid else ''} "
                        f"title_len={len(title)} abs_paras={len(abs_paras)} "
                        f"sections={len(sections)} case_sections={len(case_sections)}")

    return {
        "ok": True,
        "title": title,
        "abstract_paras": abs_paras,
        "sections": sections,
        "case_sections": case_sections,
    }

# =============================================================================
# core extractor: xml -> chunks -> candidates
# =============================================================================

def case_chunks_from_parsed(pmcid: str, parsed: Dict[str, Any], debug: bool = False) -> List[Dict[str, Any]]:
    pmcid = _normalize_pmcid(pmcid)
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

    if debug:
        log_debug(True, f"[DBG][chunks][{pmcid}] chunks={len(chunks)} "
                        f"(abs_used={min(3, len(abs_paras))}, sections_used={len(sections)})")
    return chunks

def extract_candidates_from_chunks(chunks: List[Dict[str, Any]], debug: bool = False, pmcid: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for ch in chunks:
        para = ch["text"]
        bp_spans: List[Tuple[int, int]] = []

        # BP
        for m in _RE_BP.finditer(para):
            st, ed = m.span()
            bp_spans.append((st, ed))
            ctx = sentence_first_context(para, (st, ed), clause_first=True)
            anchors = find_anchors_fixed(ctx)
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
                "label_options": label_options_from_unit("mmHg", anchors, ctx),
                "context": ctx,
            })

        # HR
        for m in _RE_HR.finditer(para):
            st, ed = m.span()
            if any((st >= bst and ed <= bed) for bst, bed in bp_spans):
                continue
            ctx = sentence_first_context(para, (st, ed), clause_first=True)
            anchors = find_anchors_fixed(ctx)
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
                "label_options": label_options_from_unit(unit, anchors, ctx),
                "context": ctx,
            })

        # generic value + unit
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
            anchors = find_anchors_fixed(ctx)

            ref_range = None
            for rm in _RE_RANGE.finditer(ctx):
                if (rm.group("unit") or "").lower() == unit_l:
                    try:
                        ref_range = {
                            "low": float(rm.group("lo")),
                            "high": float(rm.group("hi")),
                            "unit": normalize_unit(unit),
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
                "label_options": label_options_from_unit(unit, anchors, ctx),
                "context": ctx,
            }
            if ref_range:
                cand["ref_range"] = ref_range
            out.append(cand)

    if debug:
        log_debug(True, f"[DBG][extract]{'['+pmcid+']' if pmcid else ''} candidates={len(out)}")
    return out

# =============================================================================
# per-PMCID runner (resume-safe)
# =============================================================================

def per_pmcid_dir(out_root: str, pmcid: str) -> str:
    pmcid = _normalize_pmcid(pmcid)
    return os.path.join(out_root, "per_pmcid", pmcid)

def done_flag(out_root: str, pmcid: str) -> str:
    return os.path.join(per_pmcid_dir(out_root, pmcid), "DONE.json")

def is_done(out_root: str, pmcid: str) -> bool:
    return os.path.exists(done_flag(out_root, pmcid))

def mark_done(out_root: str, pmcid: str, meta: Dict[str, Any]) -> None:
    p = done_flag(out_root, pmcid)
    ensure_dir(os.path.dirname(p))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def run_one(
    pmcid: str,
    xml_text: str,
    out_root: str,
    debug: bool = False,
    debug_preview_chars: int = 220,
) -> Tuple[str, int, int, bool, str]:
    """
    Return: (pmcid, n_chunks, n_cands, skipped, err)
    """
    pmcid = _normalize_pmcid(pmcid)

    # NOTE: resume filtering ideally happens before submit(),
    # but still safe to check here
    if is_done(out_root, pmcid):
        return pmcid, 0, 0, True, ""

    perdir = per_pmcid_dir(out_root, pmcid)
    ensure_dir(perdir)

    if not (xml_text or "").strip():
        return pmcid, 0, 0, False, "empty xml"

    log_debug(debug, f"[DBG][{pmcid}] step=read_xml bytes={len(xml_text)} sha1={sha1(xml_text[:2000])}")

    parsed = parse_pmc_xml(xml_text, debug=debug, pmcid=pmcid)
    if not parsed.get("ok"):
        return pmcid, 0, 0, False, f"parse_pmc_xml failed: {parsed.get('error', '')}"

    if debug and debug_preview_chars > 0:
        t = parsed.get("title", "") or ""
        ap = parsed.get("abstract_paras") or []
        log_debug(True, f"[DBG][{pmcid}] title={t[:160]!r}")
        if ap:
            log_debug(True, f"[DBG][{pmcid}] abs0={ap[0][:debug_preview_chars]!r}")

    chunks = case_chunks_from_parsed(pmcid, parsed, debug=debug)
    cands = extract_candidates_from_chunks(chunks, debug=debug, pmcid=pmcid)

    # write per-pmcid
    write_jsonl(os.path.join(perdir, "case_chunks.jsonl"), chunks)
    write_jsonl(os.path.join(perdir, "mentions_candidates.jsonl"), cands)

    mark_done(out_root, pmcid, {
        "pmcid": pmcid,
        "ts": now_ts(),
        "n_chunks": len(chunks),
        "n_candidates": len(cands),
    })

    log_debug(debug, f"[DBG][{pmcid}] step=write_per_pmcid chunks={len(chunks)} cands={len(cands)}")
    return pmcid, len(chunks), len(cands), False, ""

# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser("rerun_phaseA_from_pmcxml (offline)")
    ap.add_argument("--run_dir", required=True, help="Stage3 run_dir, e.g. .../out_phaseA/20260126_164618")
    ap.add_argument("--out_subdir", default="phaseA_rerun", help="Output subdir under run_dir")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0 means all pmcids")
    ap.add_argument("--resume", action="store_true", help="skip pmcids with DONE.json")
    ap.add_argument("--debug", action="store_true", help="print debug logs")
    ap.add_argument("--debug_every", type=int, default=0, help="debug every N pmcids (0 means debug only first N_debug)")
    ap.add_argument("--debug_first", type=int, default=3, help="if --debug and --debug_every=0, debug first K pmcids")
    ap.add_argument("--debug_preview_chars", type=int, default=220, help="preview chars for title/abstract when debug")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    xml_dir = os.path.join(run_dir, ".cache", "pmc_xml")
    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(f"missing pmc xml cache dir: {xml_dir}")

    out_root = os.path.join(run_dir, args.out_subdir)
    out_merged = os.path.join(out_root, "merged")
    ensure_dir(out_merged)

    merged_case = os.path.join(out_merged, "case_chunks.jsonl")
    merged_mentions = os.path.join(out_merged, "mentions_candidates.jsonl")
    failures_path = os.path.join(out_root, "failures.jsonl")

    if not args.resume:
        ensure_dir(os.path.dirname(merged_case))
        open(merged_case, "w", encoding="utf-8").close()
        open(merged_mentions, "w", encoding="utf-8").close()
        if os.path.exists(failures_path):
            os.remove(failures_path)

    pmcids = load_pmcids_from_cache(xml_dir)
    if args.limit and args.limit > 0:
        pmcids = pmcids[: args.limit]

    if args.resume:
        pmcids = [p for p in pmcids if not is_done(out_root, p)]

    print(f"[INFO] xml_dir={xml_dir}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] n_pmcids={len(pmcids)} threads={args.threads} resume={bool(args.resume)}")

    import threading
    lock_merge = threading.Lock()

    ok = 0
    fail = 0
    total_chunks = 0
    total_cands = 0

    def should_debug_idx(i: int) -> bool:
        if not args.debug:
            return False
        if args.debug_every and args.debug_every > 0:
            return (i % args.debug_every) == 0
        return i < max(0, int(args.debug_first))

    pbar = tqdm(total=len(pmcids), desc="Rerun PhaseA from cached PMC XML", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=max(1, int(args.threads))) as ex:
        futs = {}
        for i, pmcid in enumerate(pmcids):
            xml_text = read_xml(xml_dir, pmcid)
            if not xml_text:
                with open(failures_path, "a", encoding="utf-8") as fo:
                    fo.write(json.dumps({"pmcid": pmcid, "stage": "read_xml", "error": "missing xml file"}, ensure_ascii=False) + "\n")
                fail += 1
                pbar.update(1)
                continue

            dbg = should_debug_idx(i)
            if dbg:
                log_debug(True, f"[DBG][{pmcid}] submit idx={i}")

            futs[ex.submit(run_one, pmcid, xml_text, out_root, dbg, int(args.debug_preview_chars))] = pmcid

        for fut in as_completed(futs):
            pmcid = futs[fut]
            try:
                pmcid, n_chunks, n_cands, skipped, err = fut.result()
                if err:
                    with open(failures_path, "a", encoding="utf-8") as fo:
                        fo.write(json.dumps({"pmcid": pmcid, "stage": "extract", "error": err}, ensure_ascii=False) + "\n")
                    fail += 1
                else:
                    if not skipped:
                        total_chunks += n_chunks
                        total_cands += n_cands

                        perdir = per_pmcid_dir(out_root, pmcid)
                        per_case = os.path.join(perdir, "case_chunks.jsonl")
                        per_mentions = os.path.join(perdir, "mentions_candidates.jsonl")

                        # merge lines (read per-pmcid output)
                        with open(per_case, "r", encoding="utf-8") as f1:
                            case_lines = f1.read().splitlines()
                        with open(per_mentions, "r", encoding="utf-8") as f2:
                            mention_lines = f2.read().splitlines()

                        with lock_merge:
                            append_lines(merged_case, case_lines)
                            append_lines(merged_mentions, mention_lines)

                    ok += 1

            except Exception as e:
                with open(failures_path, "a", encoding="utf-8") as fo:
                    fo.write(json.dumps({"pmcid": pmcid, "stage": "exception", "error": str(e)}, ensure_ascii=False) + "\n")
                fail += 1
            finally:
                pbar.update(1)

    pbar.close()

    print("[DONE] rerun finished")
    print(f"  merged_case     : {merged_case}")
    print(f"  merged_mentions : {merged_mentions}")
    print(f"  failures        : {failures_path}")
    print(f"  ok_pmcids={ok} fail_pmcids={fail} total_chunks={total_chunks} total_candidates={total_cands}")

if __name__ == "__main__":
    main()
