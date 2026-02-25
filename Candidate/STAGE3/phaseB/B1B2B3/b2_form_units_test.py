#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
b2_form_units_test.py

Test: read B1 outputs jsonl -> extract hit sentences by rules -> call LLM to form units
Output: units as plain text lines (rule-reformatted), one per unit.

REVISION (more stable / less noise):
1) Load system prompt from external file (B2.txt) (same as your version)
2) Better candidate-sentence selection:
   - First collect phrase-hit sentences
   - Then rank sentences to prefer definitional/existence statements
   - Filter/penalize "method/tool/guideline/biomarker/questionnaire" sentences
   - Fallback to first 2 sentences only if still empty
3) Add optional "approx_flag" computed by rules (no LLM change):
   - If HPO name tokens barely overlap with canonical/support -> approx_flag=1
   - You can keep it off by default to preserve exact TSV output
4) Optional: --print_meta to append confidence/note/approx_flag at end (TSV)

Usage:
  export DEEPSEEK_API_KEY="..."
  python3 b2_form_units_test.py \
    --b1_jsonl /cluster/home/gw/.../b1_outputs.jsonl \
    --n 10 \
    --base_url https://api.deepseek.com \
    --model deepseek-chat

Optional:
  --prompt_txt /path/to/B2.txt
  --print_meta  (append confidence/note/approx_flag)
"""

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

from Clients.b2_llm_client import B2LLMClient, B2LLMConfig


# ----------------------------
# sentence split (simple + robust enough for abstracts)
# ----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
_WS = re.compile(r"\s+")

DEFAULT_PROMPT_TXT = (
    "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/"
    "pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/"
    "phaseB/B1B2B3/prompts/B2.txt"
)

# penalize method/tool/guideline sentences (high precision, not exhaustive)
_METHOD_BAD_PAT = re.compile(
    r"\b("
    r"questionnaire|questionnaires|survey|score|scoring|decision-making|"
    r"guideline|guidelines|recommendation|recommended|pillar|diagnostic|diagnosis|"
    r"approach|approaches|method|methods|pipeline|assay|assays|marker|markers|biomarker|biomarkers|"
    r"culture-independent|susceptibility|testing|point-of-care|"
    r"developed|advancement|reconceptualization|emphasis"
    r")\b",
    re.IGNORECASE,
)

# prefer definitional / existence statements
_DEF_GOOD_PAT = re.compile(
    r"\b("
    r"is a|is an|are a|are an|"
    r"patients with|people with|individuals with|subjects with|"
    r"characterized by|defined by|caused by|"
    r"inherited in an|autosomal dominant|autosomal recessive|x-linked"
    r")\b",
    re.IGNORECASE,
)


def clean(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s


def split_sentences(text: str) -> List[str]:
    text = clean(text)
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    return [clean(x) for x in sents if clean(x)]


def norm_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\(\)\[\]\{\},;:\"'`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_system_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt txt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        raise ValueError(f"Prompt txt is empty: {path}")
    return txt


def sentence_score(sent: str, phrases_n: List[str]) -> float:
    """
    Heuristic ranking score:
      +10 if definitional/existence pattern
      +2  per phrase hit
      -6  if looks like method/tool/guideline sentence
      -0.005 * len(sentence) (slight length penalty)
    """
    s = clean(sent)
    sn = norm_for_match(s)

    score = 0.0
    if _DEF_GOOD_PAT.search(s):
        score += 10.0
    if _METHOD_BAD_PAT.search(s):
        score -= 6.0

    hit_cnt = 0
    for p in phrases_n:
        if p and p in sn:
            hit_cnt += 1
    score += 2.0 * hit_cnt

    score -= 0.005 * len(s)
    return score


def find_hit_sentences(abstract: str, phrases: List[str], max_sents: int = 5) -> List[str]:
    """
    Improved selection:
    1) Collect phrase-hit sentences (fuzzy contains).
    2) Rank them to prefer definitional/existence and avoid method/tool sentences.
    3) If no hit sentences, fallback to first 2 sentences.
    """
    sents = split_sentences(abstract)
    if not sents:
        return []

    phrases_n = [norm_for_match(p) for p in phrases if clean(p)]
    phrases_n = [p for p in phrases_n if p]

    hits: List[str] = []
    for sent in sents:
        sn = norm_for_match(sent)
        if any(p in sn for p in phrases_n):
            hits.append(sent)

    if not hits:
        return sents[:2]

    # rank
    scored = [(sentence_score(s, phrases_n), s) for s in hits]
    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    for _, s in scored:
        if s not in out:
            out.append(s)
        if len(out) >= max_sents:
            break
    return out


def short_def(hpo_name: str, hpo_def: str, hpo_llm_def: str, max_chars: int = 320) -> str:
    base = clean(hpo_llm_def) or clean(hpo_def) or ""
    text = f"{clean(hpo_name)}. {base}".strip()
    return text[:max_chars]


def build_user_prompt(
    hpo_id: str,
    hpo_name: str,
    hpo_def: str,
    hpo_llm_def: str,
    phrases: List[str],
    hit_sents: List[str],
) -> str:
    ddef = short_def(hpo_name, hpo_def, hpo_llm_def)

    ph = [clean(p) for p in phrases if clean(p)][:6]
    sents = [clean(s) for s in hit_sents if clean(s)][:6]

    user = (
        f"HPO_ID: {hpo_id}\n"
        f"HPO_NAME: {clean(hpo_name)}\n"
        f"DEFINITION: {ddef}\n"
        f"PHRASE_CANDIDATES:\n- " + "\n- ".join(ph) + "\n"
        f"CANDIDATE_SENTENCES:\n1) " + "\n2) ".join(sents) + "\n\n"
        "Task:\n"
        "1) Decide if any sentence supports this HPO.\n"
        "2) If supports, choose the best sentence and output a short canonical_phrase.\n"
        "3) If not, output empty canonical_phrase and support_sentence.\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "verdict": "support | not_support | unclear",\n'
        '  "canonical_phrase": "string",\n'
        '  "support_sentence": "string",\n'
        '  "confidence": 0.0,\n'
        '  "note": "brief reason"\n'
        "}"
    )
    return user


_STOPWORDS = {
    "of", "the", "and", "a", "an", "to", "in", "on", "for", "with", "without", "by",
    "abnormality", "abnormal", "inheritance", "disorder", "disease", "syndrome",
}

def approx_flag_by_overlap(hpo_name: str, canonical: str, support: str) -> int:
    """
    Rule-only flag:
      approx=1 if HPO name shares too few informative tokens with (canonical + support).
    This helps catch 'related_not_exact' cases without changing LLM.
    """
    hn = norm_for_match(hpo_name)
    cn = norm_for_match(canonical + " " + support)

    h_toks = [t for t in hn.split() if t and t not in _STOPWORDS and len(t) >= 4]
    if not h_toks:
        return 0

    hit = sum(1 for t in h_toks if t in cn)
    # require at least 1 informative token hit OR 25% hit rate
    if hit >= 1:
        return 0
    return 1


def reform_unit_plain_text(
    hpo_id: str,
    hpo_name: str,
    pmid: str,
    out: Dict[str, Any],
    print_meta: bool = False,
) -> Optional[str]:
    verdict = (out.get("verdict") or "").strip().lower()
    canonical = clean(out.get("canonical_phrase") or "")
    support = clean(out.get("support_sentence") or "")

    if verdict != "support":
        return None
    if not canonical or not support:
        return None

    support = support[:420]
    canonical = canonical[:120]

    if not print_meta:
        return f"{hpo_id}\t{clean(hpo_name)}\t{pmid}\t{canonical}\t{support}"

    # meta fields (optional)
    conf = out.get("confidence", "")
    note = clean(out.get("note") or "")
    approx = approx_flag_by_overlap(hpo_name, canonical, support)
    return f"{hpo_id}\t{clean(hpo_name)}\t{pmid}\t{canonical}\t{support}\tconf={conf}\tapprox={approx}\tnote={note}"


def read_jsonl_head(path: str, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if len(rows) >= n:
                break
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1_jsonl", required=True)
    ap.add_argument("--n", type=int, default=10)

    ap.add_argument("--base_url", type=str, default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))

    ap.add_argument("--max_sents", type=int, default=5)
    ap.add_argument("--prompt_txt", type=str, default=DEFAULT_PROMPT_TXT, help="System prompt txt path (B2.txt)")
    ap.add_argument("--print_meta", action="store_true", help="Append conf/approx/note at end of TSV")
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY env var or pass --api_key.")

    system_prompt = load_system_prompt(args.prompt_txt)

    client = B2LLMClient(B2LLMConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=0.0,
        max_tokens=512,
        timeout=60,
    ))

    rows = read_jsonl_head(args.b1_jsonl, args.n)

    for obj in rows:
        hpo_id = obj.get("hpo_id", "")
        hpo_name = obj.get("hpo_name", "")
        hpo_def = obj.get("hpo_def", "")
        hpo_llm_def = obj.get("hpo_llm_def", "")
        pmid = obj.get("pmid", obj.get("doc_id", "")) or ""
        abstract = obj.get("abstract", "") or ""

        phrases = obj.get("output_lines", None) or []
        if not isinstance(phrases, list):
            phrases = [str(phrases)]

        # RULE: only generate units when there are B1 output_lines
        if not phrases:
            continue

        hit_sents = find_hit_sentences(abstract, phrases, max_sents=args.max_sents)
        if not hit_sents:
            continue

        user_prompt = build_user_prompt(hpo_id, hpo_name, hpo_def, hpo_llm_def, phrases, hit_sents)
        out = client.chat_json(system=system_prompt, user=user_prompt)

        unit_line = reform_unit_plain_text(hpo_id, hpo_name, pmid, out, print_meta=args.print_meta)
        if unit_line:
            print(unit_line)


if __name__ == "__main__":
    main()
