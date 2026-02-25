#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pubmed_pmc_client.py

A small PubMed/PMC client (NCBI E-utilities) + a __main__ test.

What it does:
1) PubMed ESearch: term -> PMIDs
2) PubMed ELink: PMIDs -> PMCIDs (if available)
3) PMC EFetch: PMCID -> full-text XML (NXML-like)
4) Parse PMC XML into sections + paragraphs (best-effort)
5) Simple "case/clinical" section detection (rule-based)

Usage:
  python pubmed_pmc_client.py --term "Hypocalcemic seizures" --email "you@example.com" --retmax 20
  python pubmed_pmc_client.py --term "\"Congenital hypothyroidism\"" --email "you@example.com" --need-pmc
  python pubmed_pmc_client.py --pmcid "PMC1234567" --email "you@example.com"
"""

from __future__ import annotations

import argparse
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# -----------------------------
# Helpers
# -----------------------------

def _sleep_polite(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds)

def _compact_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _safe_text(el) -> str:
    # Join all nested text
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
    """
    Very simple heuristic for identifying clinical/case sections.
    """
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

def _normalize_pmcid(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return x
    if x.upper().startswith("PMC"):
        return "PMC" + re.sub(r"^PMC", "", x, flags=re.IGNORECASE)
    # allow numeric only
    if x.isdigit():
        return "PMC" + x
    return x


# -----------------------------
# Client
# -----------------------------

@dataclass
class NCBIConfig:
    email: str
    tool: str = "hpo-agent"
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 2
    polite_sleep: float = 0.34  # ~3 req/sec; adjust if you want slower


class PubMedPMCClient:
    def __init__(self, cfg: NCBIConfig):
        if not cfg.email:
            raise ValueError("NCBI requires an email parameter for polite use. Provide --email.")
        self.cfg = cfg
        self.sess = requests.Session()

    def _get(self, url: str, params: Dict[str, Any]) -> requests.Response:
        # add standard NCBI fields
        params = dict(params)
        params["tool"] = self.cfg.tool
        params["email"] = self.cfg.email
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=self.cfg.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    # backoff + retry
                    _sleep_polite(self.cfg.polite_sleep * (2 ** attempt))
                    continue
                return r
            except Exception as e:
                last_err = e
                _sleep_polite(self.cfg.polite_sleep * (2 ** attempt))
                continue
        raise RuntimeError(f"HTTP GET failed after retries: {url} err={last_err}")

    # -------- PubMed --------

    def pubmed_esearch(self, term: str, retmax: int = 20) -> Dict[str, Any]:
        url = NCBI_BASE + "esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": int(retmax),
            "retmode": "json",
        }
        r = self._get(url, params)
        if not r.ok:
            return {"ok": False, "error": f"PubMed ESearch failed: {r.status_code}", "text": r.text[:200]}
        j = r.json()
        ids = j.get("esearchresult", {}).get("idlist", []) or []
        _sleep_polite(self.cfg.polite_sleep)
        return {
            "ok": True,
            "term": term,
            "pmids": ids,
            "count": j.get("esearchresult", {}).get("count"),
            "retmax": retmax,
        }

    def pubmed_elink_to_pmc(self, pmids: List[str]) -> Dict[str, Any]:
        """
        Link PubMed -> PMC. Returns mapping pmid -> [pmcid...]
        """
        url = NCBI_BASE + "elink.fcgi"
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": ",".join(pmids),
            "retmode": "json",
            "linkname": "pubmed_pmc",
        }
        r = self._get(url, params)
        if not r.ok:
            return {"ok": False, "error": f"ELink pubmed->pmc failed: {r.status_code}", "text": r.text[:200]}

        j = r.json()
        # JSON layout: linksets[...].linksetdbs[...].links
        mapping: Dict[str, List[str]] = {}
        for ls in j.get("linksets", []) or []:
            pmid = None
            try:
                pmid = (ls.get("ids") or [None])[0]
            except Exception:
                pmid = None
            pmcids: List[str] = []
            for ldb in ls.get("linksetdbs", []) or []:
                if (ldb.get("linkname") or "").lower() == "pubmed_pmc":
                    for x in (ldb.get("links") or []):
                        if x:
                            pmcids.append(_normalize_pmcid(str(x)))
            if pmid:
                mapping[str(pmid)] = pmcids

        # also collect a flattened set
        flat: List[str] = []
        for v in mapping.values():
            for x in v:
                if x and x not in flat:
                    flat.append(x)

        _sleep_polite(self.cfg.polite_sleep)
        return {"ok": True, "mapping": mapping, "pmcids": flat}

    # -------- PMC --------

    def pmc_efetch_xml(self, pmcid: str) -> Dict[str, Any]:
        """
        Fetch PMC full text as XML (best-effort).
        """
        pmcid = _normalize_pmcid(pmcid)
        url = NCBI_BASE + "efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmcid,
            "retmode": "xml",
        }
        r = self._get(url, params)
        if not r.ok:
            return {"ok": False, "error": f"PMC EFetch failed: {r.status_code}", "text": r.text[:200], "pmcid": pmcid}
        _sleep_polite(self.cfg.polite_sleep)
        return {"ok": True, "pmcid": pmcid, "xml": r.text}

    # -------- Parse PMC XML --------

    def parse_pmc_xml(self, xml_text: str) -> Dict[str, Any]:
        """
        Best-effort parsing of PMC XML (NXML-like). Extract:
          - title
          - abstract paragraphs
          - sections -> paragraphs
          - case_sections subset (heuristic on section title)
        """
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
        except Exception as e:
            return {"ok": False, "error": f"XML parse error: {e}"}

        # Article title
        title_el = root.find(".//article-title")
        title = _safe_text(title_el)

        # Abstract paragraphs
        abs_paras: List[str] = []
        for p in root.findall(".//abstract//p"):
            t = _safe_text(p)
            if t:
                abs_paras.append(t)

        # Body sections
        sections: List[Dict[str, Any]] = []
        case_sections: List[Dict[str, Any]] = []

        # Iterate all <sec>
        for sec in root.findall(".//body//sec"):
            sec_title = _safe_text(sec.find("./title"))
            paras: List[str] = []
            for p in sec.findall(".//p"):
                t = _safe_text(p)
                if t:
                    paras.append(t)
            if not paras and not sec_title:
                continue

            item = {
                "title": sec_title,
                "n_paras": len(paras),
                "paras": paras,
            }
            sections.append(item)
            if _is_caseish_title(sec_title):
                case_sections.append(item)

        return {
            "ok": True,
            "title": title,
            "abstract_paras": abs_paras,
            "sections": sections,
            "case_sections": case_sections,
        }


# -----------------------------
# __main__ test
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", type=str, required=True, help="NCBI polite use email")
    ap.add_argument("--api-key", type=str, default=None, help="Optional NCBI API key")
    ap.add_argument("--tool", type=str, default="hpo-agent-demo")

    ap.add_argument("--term", type=str, default="", help="Search term for PubMed")
    ap.add_argument("--retmax", type=int, default=20, help="PubMed ESearch retmax")
    ap.add_argument("--need-pmc", action="store_true", help="Require that results have PMCIDs")

    ap.add_argument("--pmcid", type=str, default="", help="Directly fetch this PMCID (skips PubMed search)")

    ap.add_argument("--show-sections", type=int, default=3, help="How many sections to print (0=none)")
    ap.add_argument("--show-case-paras", type=int, default=3, help="How many paragraphs per case section to print")
    ap.add_argument("--polite-sleep", type=float, default=0.34)

    args = ap.parse_args()

    cfg = NCBIConfig(
        email=args.email,
        tool=args.tool,
        api_key=args.api_key,
        polite_sleep=float(args.polite_sleep),
    )
    cli = PubMedPMCClient(cfg)

    # If PMCID provided: fetch + parse directly
    if args.pmcid.strip():
        pmcid = _normalize_pmcid(args.pmcid)
        print(f"[TEST] Fetch PMC full text: {pmcid}")
        r = cli.pmc_efetch_xml(pmcid)
        if not r.get("ok"):
            print("[ERR]", r)
            return
        parsed = cli.parse_pmc_xml(r["xml"])
        if not parsed.get("ok"):
            print("[ERR]", parsed)
            return

        print("\n=== ARTICLE TITLE ===")
        print(parsed.get("title", ""))

        if parsed.get("abstract_paras"):
            print("\n=== ABSTRACT (first 2 paras) ===")
            for p in parsed["abstract_paras"][:2]:
                print("-", p[:500])

        print(f"\n=== SECTIONS total={len(parsed.get('sections', []))} case_like={len(parsed.get('case_sections', []))} ===")

        if args.show_sections > 0:
            for i, sec in enumerate(parsed["sections"][:args.show_sections], start=1):
                print(f"\n[Section {i}] {sec.get('title','') or '(no title)'}  paras={sec['n_paras']}")
                if sec["paras"]:
                    print("  -", sec["paras"][0][:400])

        if parsed["case_sections"]:
            print("\n=== CASE-LIKE SECTIONS (heuristic) ===")
            for i, sec in enumerate(parsed["case_sections"], start=1):
                print(f"\n[CaseSec {i}] {sec.get('title','') or '(no title)'}  paras={sec['n_paras']}")
                for p in sec["paras"][:max(0, int(args.show_case_paras))]:
                    print("  -", p[:500])

        return

    # Otherwise: PubMed search -> pmids -> pmcids -> fetch first pmcid
    term = args.term.strip()
    if not term:
        raise ValueError("Provide --term for PubMed search, or --pmcid to fetch directly.")

    print(f"[TEST] PubMed ESearch term={term!r} retmax={args.retmax}")
    sr = cli.pubmed_esearch(term, retmax=args.retmax)
    if not sr.get("ok"):
        print("[ERR]", sr)
        return
    pmids = sr["pmids"]
    print(f"[OK] pmids={len(pmids)}  sample={pmids[:5]}")

    if not pmids:
        print("[DONE] No PubMed hits.")
        return

    print("\n[TEST] PubMed -> PMC ELink")
    lr = cli.pubmed_elink_to_pmc(pmids)
    if not lr.get("ok"):
        print("[ERR]", lr)
        return
    pmcids = lr["pmcids"]
    print(f"[OK] pmcids={len(pmcids)}  sample={pmcids[:10]}")

    if args.need_pmc and not pmcids:
        print("[DONE] No PMC hits (need PMC). Try different term or remove --need-pmc.")
        return
    if not pmcids:
        print("[DONE] No PMCIDs found; cannot fetch full text via PMC.")
        return

    pmcid0 = pmcids[0]
    print(f"\n[TEST] Fetch PMC full text for first PMCID: {pmcid0}")
    fr = cli.pmc_efetch_xml(pmcid0)
    if not fr.get("ok"):
        print("[ERR]", fr)
        return

    parsed = cli.parse_pmc_xml(fr["xml"])
    if not parsed.get("ok"):
        print("[ERR]", parsed)
        return

    print("\n=== ARTICLE TITLE ===")
    print(parsed.get("title", ""))

    if parsed.get("abstract_paras"):
        print("\n=== ABSTRACT (first 2 paras) ===")
        for p in parsed["abstract_paras"][:2]:
            print("-", p[:500])

    print(f"\n=== SECTIONS total={len(parsed.get('sections', []))} case_like={len(parsed.get('case_sections', []))} ===")

    if parsed["case_sections"]:
        print("\n=== CASE-LIKE SECTIONS (heuristic) ===")
        for i, sec in enumerate(parsed["case_sections"][:3], start=1):
            print(f"\n[CaseSec {i}] {sec.get('title','') or '(no title)'}  paras={sec['n_paras']}")
            for p in sec["paras"][:max(0, int(args.show_case_paras))]:
                print("  -", p[:500])
    else:
        # fallback: show a few normal sections
        if args.show_sections > 0:
            for i, sec in enumerate(parsed["sections"][:args.show_sections], start=1):
                print(f"\n[Section {i}] {sec.get('title','') or '(no title)'}  paras={sec['n_paras']}")
                if sec["paras"]:
                    print("  -", sec["paras"][0][:400])


if __name__ == "__main__":
    main()
