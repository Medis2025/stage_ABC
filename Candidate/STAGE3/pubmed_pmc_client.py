#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pubmed_pmc_client.py  (REVISED to support PhaseA/Stage3 runner without breaking old calls)

What changed vs your current file
================================
✅ DOES NOT change any existing public methods/signatures you already have:
  - pubmed_esearch(term, retmax)
  - pubmed_elink_to_pmc(pmids)
  - pubmed_elink_to_pmc_batch(pmids, batch_size)
  - pmc_efetch_xml(pmcid)
  - pmc_efetch_xml_batch(pmcids, batch_size)
  - parse_pmc_xml(xml_text)

✅ ADDS the PubMed EFetch methods your PhaseA runner is calling:
  - pubmed_efetch_xml(pmid)                     # single PMID -> PubMedArticle XML
  - pubmed_efetch_xml_batch(pmids, batch_size)  # batch PMIDs -> {pmid: xml}

✅ Adds optional on-disk cache for PubMed efetch XML (similar to PMC xml cache)
✅ Adds helper parse_pubmed_xml_abstracts() so PhaseA can build "weak evidence pool" robustly

Why you needed this
===================
Your runner error:
  'PubMedPMCClient' object has no attribute 'pubmed_efetch_xml_batch'

This file implements it, keeping everything else compatible.

PhaseA-style usage (typical)
============================
cfg = NCBIConfig(
  email="you@x.com",
  api_key="optional",
  tool="hpo-agent",
  timeout=30,
  max_retries=2,
  polite_sleep=0.0,
  cache_dir="/path/to/cache",
  cache_esearch=True,
  cache_elink=True,
  cache_pmc_xml=True,
  cache_pubmed_xml=True,   # NEW
)

cli = PubMedPMCClient(cfg)

# 1) term -> pmids
sr = cli.pubmed_esearch(term, retmax=20)
pmids = sr["pmids"]

# 2) pmids -> pubmed efetch xml (abstracts)
xr = cli.pubmed_efetch_xml_batch(pmids, batch_size=200)
pmid_to_xml = xr["pmid_to_xml"]

# 3) parse abstracts into evidence rows
rows = cli.parse_pubmed_xml_abstracts(pmid_to_xml)  # -> list[dict]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import requests


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# -----------------------------
# Helpers
# -----------------------------

def _sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds)

def _compact_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

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

def _normalize_pmcid(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return x
    if x.upper().startswith("PMC"):
        return "PMC" + re.sub(r"^PMC", "", x, flags=re.IGNORECASE)
    if x.isdigit():
        return "PMC" + x
    return x

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    n = max(1, int(n))
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# -----------------------------
# Config
# -----------------------------

@dataclass
class NCBIConfig:
    email: str
    tool: str = "hpo-agent"
    api_key: Optional[str] = None

    timeout: float = 30.0
    max_retries: int = 2

    # NOTE: if your runner does global throttling, set polite_sleep=0.0
    polite_sleep: float = 0.0

    # Optional caching
    cache_dir: str = ""                 # if empty => no disk cache
    cache_esearch: bool = False
    cache_elink: bool = False
    cache_pmc_xml: bool = False

    # NEW: PubMed EFetch (PMID XML) cache
    cache_pubmed_xml: bool = False


# -----------------------------
# Client
# -----------------------------

class PubMedPMCClient:
    def __init__(self, cfg: NCBIConfig):
        if not cfg.email:
            raise ValueError("NCBI requires an email parameter for polite use. Provide --email.")
        self.cfg = cfg
        self.sess = requests.Session()

        self.cache_dir = (cfg.cache_dir or "").strip()
        if self.cache_dir:
            _ensure_dir(self.cache_dir)
            _ensure_dir(os.path.join(self.cache_dir, "esearch"))
            _ensure_dir(os.path.join(self.cache_dir, "elink"))
            _ensure_dir(os.path.join(self.cache_dir, "pmc_xml"))
            _ensure_dir(os.path.join(self.cache_dir, "pubmed_xml"))  # NEW

    # -------------------------
    # Low-level HTTP
    # -------------------------

    def _get(self, url: str, params: Dict[str, Any]) -> requests.Response:
        params = dict(params)
        params["tool"] = self.cfg.tool
        params["email"] = self.cfg.email
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key

        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=self.cfg.timeout)

                # Retry transient statuses
                if r.status_code in (429, 500, 502, 503, 504):
                    _sleep(self.cfg.polite_sleep * (2 ** attempt) if self.cfg.polite_sleep else (0.25 * (2 ** attempt)))
                    continue

                return r

            except Exception as e:
                last_err = e
                _sleep(self.cfg.polite_sleep * (2 ** attempt) if self.cfg.polite_sleep else (0.25 * (2 ** attempt)))
                continue

        raise RuntimeError(f"HTTP GET failed after retries: {url} err={last_err}")

    # -------------------------
    # Cache helpers
    # -------------------------

    def _cache_path(self, kind: str, key: str, ext: str) -> str:
        # kind in {"esearch","elink","pmc_xml","pubmed_xml"}
        base = os.path.join(self.cache_dir, kind)
        return os.path.join(base, f"{key}.{ext}")

    def _cache_read_json(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        p = self._cache_path(kind, key, "json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _cache_write_json(self, kind: str, key: str, obj: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        p = self._cache_path(kind, key, "json")
        tmp = p + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            os.replace(tmp, p)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _cache_read_text(self, kind: str, key: str, ext: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        p = self._cache_path(kind, key, ext)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def _cache_write_text(self, kind: str, key: str, ext: str, text: str) -> None:
        if not self.cache_dir:
            return
        p = self._cache_path(kind, key, ext)
        tmp = p + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text or "")
            os.replace(tmp, p)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    # -------------------------
    # PubMed ESearch
    # -------------------------

    def pubmed_esearch(self, term: str, retmax: int = 20) -> Dict[str, Any]:
        """
        term -> PMIDs (list)
        Cache key uses SHA1(term|retmax)
        """
        term = term or ""
        retmax_i = int(retmax)

        cache_key = _sha1(f"{term}\n{retmax_i}")
        if self.cfg.cache_esearch and self.cache_dir:
            hit = self._cache_read_json("esearch", cache_key)
            if hit and hit.get("ok") and hit.get("term") == term and int(hit.get("retmax", -1)) == retmax_i:
                return hit

        url = NCBI_BASE + "esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax_i,
            "retmode": "json",
        }

        r = self._get(url, params)
        if not r.ok:
            out = {"ok": False, "error": f"PubMed ESearch failed: {r.status_code}", "text": r.text[:200], "term": term, "retmax": retmax_i}
            if self.cfg.cache_esearch and self.cache_dir:
                self._cache_write_json("esearch", cache_key, out)
            return out

        try:
            j = r.json()
        except Exception:
            out = {"ok": False, "error": "PubMed ESearch JSON decode failed", "text": r.text[:200], "term": term, "retmax": retmax_i}
            if self.cfg.cache_esearch and self.cache_dir:
                self._cache_write_json("esearch", cache_key, out)
            return out

        ids = j.get("esearchresult", {}).get("idlist", []) or []
        out = {
            "ok": True,
            "term": term,
            "pmids": [str(x) for x in ids if x],
            "count": j.get("esearchresult", {}).get("count"),
            "retmax": retmax_i,
        }

        if self.cfg.cache_esearch and self.cache_dir:
            self._cache_write_json("esearch", cache_key, out)
        return out

    # -------------------------
    # NEW: PubMed EFetch: PMID -> XML (abstracts / PubmedArticle)
    # -------------------------

    def pubmed_efetch_xml(self, pmid: str) -> Dict[str, Any]:
        """
        Single PMID efetch (db=pubmed, retmode=xml).
        Returns: {ok, pmid, xml, cached?}
        """
        pmid = str(pmid or "").strip()
        if not pmid:
            return {"ok": False, "error": "Empty PMID", "pmid": pmid}

        if self.cfg.cache_pubmed_xml and self.cache_dir:
            hit = self._cache_read_text("pubmed_xml", pmid, "xml")
            if hit:
                return {"ok": True, "pmid": pmid, "xml": hit, "cached": True}

        url = NCBI_BASE + "efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        r = self._get(url, params)
        if not r.ok:
            return {"ok": False, "error": f"PubMed EFetch failed: {r.status_code}", "text": r.text[:200], "pmid": pmid}

        xml_text = r.text or ""
        if self.cfg.cache_pubmed_xml and self.cache_dir:
            self._cache_write_text("pubmed_xml", pmid, "xml", xml_text)

        return {"ok": True, "pmid": pmid, "xml": xml_text, "cached": False}

    def pubmed_efetch_xml_batch(self, pmids: List[str], batch_size: int = 200) -> Dict[str, Any]:
        """
        Batch PubMed EFetch.
        Returns:
          {
            ok: True,
            pmid_to_xml: {PMID: xml_text_for_one_article_or_fallback},
            n_batches: int,
            batch_size: int,
            n_cached: int,
            n_fetched: int,
          }

        Notes:
        - For cached PMIDs, no network call.
        - For non-cached, does efetch.fcgi with id=PMID1,PMID2,... (batch).
        - Attempts to split the response (<PubmedArticleSet> with many <PubmedArticle>) into per-PMID XML.
          If splitting fails for a PMID in the batch, stores the full batch XML as fallback.
        """
        pmids_in = [str(x) for x in (pmids or []) if str(x).strip()]
        # de-dup preserve order
        pmids_uniq = list(dict.fromkeys(pmids_in))
        if not pmids_uniq:
            return {"ok": True, "pmid_to_xml": {}, "n_batches": 0, "batch_size": int(batch_size), "n_cached": 0, "n_fetched": 0}

        pmid_to_xml: Dict[str, str] = {}
        n_cached = 0
        n_fetched = 0

        remain: List[str] = []
        if self.cfg.cache_pubmed_xml and self.cache_dir:
            for pmid in pmids_uniq:
                hit = self._cache_read_text("pubmed_xml", pmid, "xml")
                if hit:
                    pmid_to_xml[pmid] = hit
                    n_cached += 1
                else:
                    remain.append(pmid)
        else:
            remain = pmids_uniq

        n_batches = 0
        for chunk in _chunks(remain, int(batch_size)):
            n_batches += 1
            url = NCBI_BASE + "efetch.fcgi"
            params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}

            r = self._get(url, params)
            if not r.ok:
                return {
                    "ok": False,
                    "error": f"PubMed batch EFetch failed: {r.status_code}",
                    "text": r.text[:200],
                    "failed_batch": chunk,
                    "n_batches": n_batches,
                }

            xml_text = r.text or ""
            extracted = self._split_batch_pubmed_xml(xml_text)

            for pmid in chunk:
                one_xml = extracted.get(pmid)
                if one_xml:
                    pmid_to_xml[pmid] = one_xml
                    n_fetched += 1
                    if self.cfg.cache_pubmed_xml and self.cache_dir:
                        self._cache_write_text("pubmed_xml", pmid, "xml", one_xml)
                else:
                    # fallback: store full batch xml so downstream can still try parsing
                    pmid_to_xml[pmid] = xml_text
                    n_fetched += 1
                    if self.cfg.cache_pubmed_xml and self.cache_dir:
                        self._cache_write_text("pubmed_xml", pmid, "xml", xml_text)

        return {
            "ok": True,
            "pmid_to_xml": pmid_to_xml,
            "n_batches": n_batches,
            "batch_size": int(batch_size),
            "n_cached": n_cached,
            "n_fetched": n_fetched,
        }

    def _split_batch_pubmed_xml(self, xml_text: str) -> Dict[str, str]:
        """
        Best-effort splitter for efetch(pubmed, id=PMID1,PMID2,...).
        Returns pmid->xml_text containing ONLY that PMID's <PubmedArticle> subtree (preferred).

        If parsing fails, returns {} and caller can fallback.
        """
        out: Dict[str, str] = {}
        if not xml_text:
            return out

        # Parse XML and extract <PubmedArticle> nodes
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            # The container is typically <PubmedArticleSet>
            # Articles are <PubmedArticle> (sometimes <PubmedBookArticle>)
            for art in root.findall(".//PubmedArticle"):
                pmid_el = art.find(".//MedlineCitation/PMID")
                pmid = _safe_text(pmid_el)
                if pmid:
                    out[str(pmid)] = ET.tostring(art, encoding="unicode")
            # Also handle PubmedBookArticle
            for art in root.findall(".//PubmedBookArticle"):
                pmid_el = art.find(".//BookDocument/PMID")
                pmid = _safe_text(pmid_el)
                if pmid:
                    out[str(pmid)] = ET.tostring(art, encoding="unicode")
            return out
        except Exception:
            return out

    # -------------------------
    # PubMed ELink: PMID -> PMCID
    # -------------------------

    def pubmed_elink_to_pmc(self, pmids: List[str]) -> Dict[str, Any]:
        """
        Backward-compatible single call. For large pmids, prefer pubmed_elink_to_pmc_batch().
        Returns:
          {
            ok: True,
            mapping: {pmid: [pmcid...]},
            pmcids:  [unique pmcids...]
          }
        """
        return self._pubmed_elink_to_pmc_one(pmids)

    def pubmed_elink_to_pmc_batch(self, pmids: List[str], batch_size: int = 200) -> Dict[str, Any]:
        """
        Batch ELink for speed + stability.

        Returns:
          {
            ok: True,
            mapping: {pmid: [pmcid...]},     # merged across batches
            pmcids:  [unique pmcids...],
            n_batches: int,
            batch_size: int,
          }
        """
        pmids_in = [str(x) for x in (pmids or []) if str(x).strip()]
        if not pmids_in:
            return {"ok": True, "mapping": {}, "pmcids": [], "n_batches": 0, "batch_size": int(batch_size)}

        merged_map: Dict[str, List[str]] = {}
        flat: List[str] = []
        n_batches = 0

        for chunk in _chunks(pmids_in, int(batch_size)):
            n_batches += 1
            r = self._pubmed_elink_to_pmc_one(chunk)

            if not r.get("ok"):
                return {"ok": False, "error": r.get("error"), "text": r.get("text", ""), "failed_batch": chunk, "n_batches": n_batches}

            mapping = r.get("mapping") or {}
            for pmid, pmc_list in mapping.items():
                if pmid not in merged_map:
                    merged_map[pmid] = []
                for pmcid in (pmc_list or []):
                    if pmcid and pmcid not in merged_map[pmid]:
                        merged_map[pmid].append(pmcid)

            for pmcid in (r.get("pmcids") or []):
                if pmcid and pmcid not in flat:
                    flat.append(pmcid)

        return {"ok": True, "mapping": merged_map, "pmcids": flat, "n_batches": n_batches, "batch_size": int(batch_size)}

    def _pubmed_elink_to_pmc_one(self, pmids: List[str]) -> Dict[str, Any]:
        """
        Internal: one ELink call on a single batch of PMIDs.
        Cache key uses SHA1(sorted(pmids)).
        """
        pmids = [str(x) for x in (pmids or []) if str(x).strip()]
        if not pmids:
            return {"ok": True, "mapping": {}, "pmcids": []}

        cache_key = _sha1(",".join(sorted(pmids)))
        if self.cfg.cache_elink and self.cache_dir:
            hit = self._cache_read_json("elink", cache_key)
            if hit and hit.get("ok") and isinstance(hit.get("mapping"), dict):
                return hit

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
            out = {"ok": False, "error": f"ELink pubmed->pmc failed: {r.status_code}", "text": r.text[:200]}
            if self.cfg.cache_elink and self.cache_dir:
                self._cache_write_json("elink", cache_key, out)
            return out

        try:
            j = r.json()
        except Exception:
            out = {"ok": False, "error": "ELink JSON decode failed", "text": r.text[:200]}
            if self.cfg.cache_elink and self.cache_dir:
                self._cache_write_json("elink", cache_key, out)
            return out

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

        flat: List[str] = []
        for v in mapping.values():
            for x in v:
                if x and x not in flat:
                    flat.append(x)

        out = {"ok": True, "mapping": mapping, "pmcids": flat}
        if self.cfg.cache_elink and self.cache_dir:
            self._cache_write_json("elink", cache_key, out)
        return out

    # -------------------------
    # PMC EFetch: PMCID -> XML
    # -------------------------

    def pmc_efetch_xml(self, pmcid: str) -> Dict[str, Any]:
        """
        Backward-compatible: single PMCID efetch.
        """
        pmcid = _normalize_pmcid(pmcid)
        if not pmcid:
            return {"ok": False, "error": "Empty PMCID", "pmcid": pmcid}

        if self.cfg.cache_pmc_xml and self.cache_dir:
            hit = self._cache_read_text("pmc_xml", pmcid, "xml")
            if hit:
                return {"ok": True, "pmcid": pmcid, "xml": hit, "cached": True}

        url = NCBI_BASE + "efetch.fcgi"
        params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
        r = self._get(url, params)
        if not r.ok:
            return {"ok": False, "error": f"PMC EFetch failed: {r.status_code}", "text": r.text[:200], "pmcid": pmcid}

        xml_text = r.text or ""

        if self.cfg.cache_pmc_xml and self.cache_dir:
            self._cache_write_text("pmc_xml", pmcid, "xml", xml_text)

        return {"ok": True, "pmcid": pmcid, "xml": xml_text, "cached": False}

    def pmc_efetch_xml_batch(self, pmcids: List[str], batch_size: int = 20) -> Dict[str, Any]:
        """
        Batch PMC EFetch.
        Returns:
          {
            ok: True,
            pmcid_to_xml: {PMCID: xml_text},
            n_batches: int,
            batch_size: int,
            n_cached: int,
            n_fetched: int,
          }
        """
        pmcids_in = [_normalize_pmcid(str(x)) for x in (pmcids or [])]
        pmcids_in = [x for x in pmcids_in if x]
        pmcids_uniq = list(dict.fromkeys(pmcids_in))
        if not pmcids_uniq:
            return {"ok": True, "pmcid_to_xml": {}, "n_batches": 0, "batch_size": int(batch_size), "n_cached": 0, "n_fetched": 0}

        pmcid_to_xml: Dict[str, str] = {}
        n_cached = 0
        n_fetched = 0

        remain: List[str] = []
        if self.cfg.cache_pmc_xml and self.cache_dir:
            for pmcid in pmcids_uniq:
                hit = self._cache_read_text("pmc_xml", pmcid, "xml")
                if hit:
                    pmcid_to_xml[pmcid] = hit
                    n_cached += 1
                else:
                    remain.append(pmcid)
        else:
            remain = pmcids_uniq

        n_batches = 0
        for chunk in _chunks(remain, int(batch_size)):
            n_batches += 1
            url = NCBI_BASE + "efetch.fcgi"
            params = {"db": "pmc", "id": ",".join(chunk), "retmode": "xml"}

            r = self._get(url, params)
            if not r.ok:
                return {
                    "ok": False,
                    "error": f"PMC batch EFetch failed: {r.status_code}",
                    "text": r.text[:200],
                    "failed_batch": chunk,
                    "n_batches": n_batches,
                }

            xml_text = r.text or ""
            extracted = self._split_batch_pmc_xml(xml_text)

            for pmcid in chunk:
                one_xml = extracted.get(pmcid)
                if one_xml:
                    pmcid_to_xml[pmcid] = one_xml
                    n_fetched += 1
                    if self.cfg.cache_pmc_xml and self.cache_dir:
                        self._cache_write_text("pmc_xml", pmcid, "xml", one_xml)
                else:
                    pmcid_to_xml[pmcid] = xml_text
                    n_fetched += 1
                    if self.cfg.cache_pmc_xml and self.cache_dir:
                        self._cache_write_text("pmc_xml", pmcid, "xml", xml_text)

        return {
            "ok": True,
            "pmcid_to_xml": pmcid_to_xml,
            "n_batches": n_batches,
            "batch_size": int(batch_size),
            "n_cached": n_cached,
            "n_fetched": n_fetched,
        }

    def _split_batch_pmc_xml(self, xml_text: str) -> Dict[str, str]:
        """
        Best-effort splitter for efetch(pmc, id=PMC1,PMC2,...).
        Returns pmcid->article_xml_text (string).
        """
        out: Dict[str, str] = {}
        if not xml_text:
            return out

        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            for art in root.findall(".//article"):
                pmc_id = None
                for aid in art.findall(".//article-id"):
                    if (aid.get("pub-id-type") or "").lower() == "pmc":
                        pmc_id = _normalize_pmcid(_safe_text(aid))
                        break
                if pmc_id:
                    out[pmc_id] = ET.tostring(art, encoding="unicode")
            return out
        except Exception:
            pass

        try:
            articles = re.findall(r"(<article\b.*?</article>)", xml_text, flags=re.IGNORECASE | re.DOTALL)
            for art_xml in articles:
                m = re.search(r'<article-id[^>]*pub-id-type="pmc"[^>]*>\s*(PMC\d+)\s*</article-id>', art_xml, flags=re.IGNORECASE)
                if m:
                    out[_normalize_pmcid(m.group(1))] = art_xml
        except Exception:
            return out

        return out

    # -------------------------
    # Parse PMC XML
    # -------------------------

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

        title_el = root.find(".//article-title")
        title = _safe_text(title_el)

        abs_paras: List[str] = []
        for p in root.findall(".//abstract//p"):
            t = _safe_text(p)
            if t:
                abs_paras.append(t)

        sections: List[Dict[str, Any]] = []
        case_sections: List[Dict[str, Any]] = []

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

        return {
            "ok": True,
            "title": title,
            "abstract_paras": abs_paras,
            "sections": sections,
            "case_sections": case_sections,
        }

    # -------------------------
    # NEW: Parse single PubMed XML (for PhaseA per-PMID usage)
    # -------------------------

    def parse_pubmed_xml(self, xml_text: str) -> Dict[str, Any]:
        """
        Backward-compatible helper expected by Stage3/PhaseA:
        parse one PubMed XML blob (either <PubmedArticle> or <PubmedArticleSet>)
        and return {ok, pmid, title, abstract, journal, year, pub_year}.
        """
        xml_text = xml_text or ""
        if not xml_text.strip():
            return {"ok": False, "error": "empty xml_text"}

        row = self._extract_one_pubmed_article_fields(xml_text, want_pmid=None)
        if not row:
            return {"ok": False, "error": "no PubmedArticle found"}

        year = row.get("year")
        return {
            "ok": True,
            "pmid": row.get("pmid", ""),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "journal": row.get("journal", ""),
            "year": year,
            "pub_year": year,
        }

    # -------------------------
    # NEW: Parse PubMed XML abstracts (for PhaseA weak evidence pool)
    # -------------------------

    def parse_pubmed_xml_abstracts(self, pmid_to_xml: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Convert pmid->(PubmedArticle XML text OR batch fallback XML) into normalized rows:
          {
            "pmid": str,
            "title": str,
            "abstract": str,     # concatenated
            "journal": str,
            "year": str,
          }

        This is designed for PhaseA Stage3 "weak evidence" building.
        It is best-effort: if the stored XML is a batch container, we try to parse and extract the correct PMID.
        """
        rows: List[Dict[str, Any]] = []
        if not pmid_to_xml:
            return rows

        # We parse each xml independently (safe, simple). If you want faster, parse batch containers once upstream.
        for pmid, xml_text in pmid_to_xml.items():
            pmid = str(pmid or "").strip()
            xml_text = xml_text or ""
            if not pmid or not xml_text.strip():
                continue

            one = self._extract_one_pubmed_article_fields(xml_text, want_pmid=pmid)
            if one:
                rows.append(one)

        return rows

    def _extract_one_pubmed_article_fields(self, xml_text: str, want_pmid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Try to parse xml_text which can be:
          - <PubmedArticle>...</PubmedArticle>
          - OR a container <PubmedArticleSet>...</PubmedArticleSet>

        If want_pmid is given and xml is a container, selects that PMID if present.
        """
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        # If root is PubmedArticle, normalize to list
        articles: List[Any] = []
        if root.tag.endswith("PubmedArticle"):
            articles = [root]
        else:
            articles = root.findall(".//PubmedArticle") or []

        chosen = None
        if want_pmid:
            for art in articles:
                pmid_el = art.find(".//MedlineCitation/PMID")
                pmid = _safe_text(pmid_el)
                if pmid and pmid == str(want_pmid):
                    chosen = art
                    break
        if chosen is None and articles:
            chosen = articles[0]
        if chosen is None:
            return None

        pmid = _safe_text(chosen.find(".//MedlineCitation/PMID"))
        title = _safe_text(chosen.find(".//Article/ArticleTitle"))
        journal = _safe_text(chosen.find(".//Article/Journal/Title"))

        year = ""
        y_el = chosen.find(".//Article/Journal/JournalIssue/PubDate/Year")
        if y_el is not None:
            year = _safe_text(y_el)
        if not year:
            # sometimes MedlineDate like "2019 Jan-Feb"
            md_el = chosen.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
            year = _safe_text(md_el)[:4] if md_el is not None else ""

        abs_parts: List[str] = []
        for a in chosen.findall(".//Article/Abstract/AbstractText"):
            t = _safe_text(a)
            if t:
                abs_parts.append(t)
        abstract = _compact_ws(" ".join(abs_parts))

        if want_pmid and pmid and pmid != str(want_pmid):
            # container parsed but did not match requested PMID
            # we still return it only if want_pmid is absent; otherwise skip
            return None

        return {
            "pmid": pmid or (str(want_pmid) if want_pmid else ""),
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
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
    ap.add_argument("--batch-elink", action="store_true", help="Use batch ELink demo")
    ap.add_argument("--batch-efetch", action="store_true", help="Use batch PMC EFetch demo")

    # NEW: PubMed EFetch demo
    ap.add_argument("--pubmed-efetch", action="store_true", help="Also do PubMed efetch XML for the returned PMIDs")
    ap.add_argument("--pubmed-efetch-batch-size", type=int, default=200)

    ap.add_argument("--cache-dir", type=str, default="", help="Optional cache directory")
    ap.add_argument("--cache-esearch", action="store_true")
    ap.add_argument("--cache-elink", action="store_true")
    ap.add_argument("--cache-pmc-xml", action="store_true")
    ap.add_argument("--cache-pubmed-xml", action="store_true")  # NEW

    ap.add_argument("--polite-sleep", type=float, default=0.0)
    ap.add_argument("--show-sections", type=int, default=3, help="How many sections to print (0=none)")
    ap.add_argument("--show-case-paras", type=int, default=3, help="How many paragraphs per case section to print")

    args = ap.parse_args()

    cfg = NCBIConfig(
        email=args.email,
        tool=args.tool,
        api_key=args.api_key,
        polite_sleep=float(args.polite_sleep),
        cache_dir=args.cache_dir,
        cache_esearch=bool(args.cache_esearch),
        cache_elink=bool(args.cache_elink),
        cache_pmc_xml=bool(args.cache_pmc_xml),
        cache_pubmed_xml=bool(args.cache_pubmed_xml),
    )
    cli = PubMedPMCClient(cfg)

    # Direct PMCID path
    if args.pmcid.strip():
        pmcid = _normalize_pmcid(args.pmcid)
        print(f"[TEST] Fetch PMC full text: {pmcid}")
        fr = cli.pmc_efetch_xml(pmcid)
        if not fr.get("ok"):
            print("[ERR]", fr)
            return
        parsed = cli.parse_pmc_xml(fr["xml"])
        if not parsed.get("ok"):
            print("[ERR]", parsed)
            return

        print("\n=== ARTICLE TITLE ===")
        print(parsed.get("title", ""))

        print(f"\n=== SECTIONS total={len(parsed.get('sections', []))} case_like={len(parsed.get('case_sections', []))} ===")
        if parsed.get("case_sections"):
            print("\n=== CASE-LIKE SECTIONS ===")
            for i, sec in enumerate(parsed["case_sections"][:3], start=1):
                print(f"\n[CaseSec {i}] {sec.get('title','') or '(no title)'}  paras={sec['n_paras']}")
                for p in sec["paras"][:max(0, int(args.show_case_paras))]:
                    print("  -", p[:500])
        return

    # PubMed search path
    term = args.term.strip()
    if not term:
        raise ValueError("Provide --term for PubMed search, or --pmcid to fetch directly.")

    print(f"[TEST] PubMed ESearch term={term!r} retmax={args.retmax}")
    sr = cli.pubmed_esearch(term, retmax=args.retmax)
    if not sr.get("ok"):
        print("[ERR]", sr)
        return
    pmids = sr["pmids"]
    print(f"[OK] pmids={len(pmids)} sample={pmids[:5]}")

    if not pmids:
        print("[DONE] No PubMed hits.")
        return

    if args.pubmed_efetch:
        print(f"\n[TEST] PubMed EFetch XML batch pmids={len(pmids)} bs={args.pubmed_efetch_batch_size}")
        xr = cli.pubmed_efetch_xml_batch(pmids, batch_size=int(args.pubmed_efetch_batch_size))
        if not xr.get("ok"):
            print("[ERR]", xr)
            return
        pmid_to_xml = xr.get("pmid_to_xml") or {}
        rows = cli.parse_pubmed_xml_abstracts(pmid_to_xml)
        print(f"[OK] parsed rows={len(rows)} sample:")
        for r in rows[:3]:
            print(" -", r.get("pmid"), "|", (r.get("title") or "")[:120], "| abs_len=", len(r.get("abstract") or ""))

    print("\n[TEST] PubMed -> PMC ELink")
    if args.batch_elink:
        lr = cli.pubmed_elink_to_pmc_batch(pmids, batch_size=200)
    else:
        lr = cli.pubmed_elink_to_pmc(pmids)

    if not lr.get("ok"):
        print("[ERR]", lr)
        return

    pmcids = lr.get("pmcids") or []
    print(f"[OK] pmcids={len(pmcids)} sample={pmcids[:10]}")

    if args.need_pmc and not pmcids:
        print("[DONE] No PMC hits (need PMC). Try different term or remove --need-pmc.")
        return
    if not pmcids:
        print("[DONE] No PMCIDs found; cannot fetch full text via PMC.")
        return

    # Fetch first (or batch fetch) PMCIDs
    if args.batch_efetch:
        batch = pmcids[:5]
        print(f"\n[TEST] Batch EFetch PMC full text for: {batch}")
        br = cli.pmc_efetch_xml_batch(batch, batch_size=5)
        if not br.get("ok"):
            print("[ERR]", br)
            return
        pmcid_to_xml = br.get("pmcid_to_xml") or {}
        pmcid0 = batch[0]
        xml0 = pmcid_to_xml.get(pmcid0, "")
    else:
        pmcid0 = pmcids[0]
        print(f"\n[TEST] Fetch PMC full text for first PMCID: {pmcid0}")
        fr = cli.pmc_efetch_xml(pmcid0)
        if not fr.get("ok"):
            print("[ERR]", fr)
            return
        xml0 = fr["xml"]

    parsed = cli.parse_pmc_xml(xml0)
    if not parsed.get("ok"):
        print("[ERR]", parsed)
        return

    print("\n=== ARTICLE TITLE ===")
    print(parsed.get("title", ""))

    print(f"\n=== SECTIONS total={len(parsed.get('sections', []))} case_like={len(parsed.get('case_sections', []))} ===")
    if parsed.get("case_sections"):
        print("\n=== CASE-LIKE SECTIONS ===")
        for i, sec in enumerate(parsed["case_sections"][:3], start=1):
            print(f"\n[CaseSec {i}] {sec.get('title','') or '(no title)'}  paras={sec['n_paras']}")
            for p in sec["paras"][:max(0, int(args.show_case_paras))]:
                print("  -", p[:500])


if __name__ == "__main__":
    main()
