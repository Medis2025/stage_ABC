#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_and_eval_stage1.py  (FIXED: _chunk_list + small robustness)

Fixes
- ✅ Define _chunk_list() (your crash: NameError)
- ✅ Minor safety: handle empty concat_texts / empty phrases
- ✅ Keep everything else the same

See "Run example" at bottom.
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set, Iterable

from tqdm import tqdm

import numpy as np
import torch

# ---- Try FAISS (fast); fallback to sklearn (slow)

import faiss  
_HAVE_FAISS = True
# except Exception:
#     faiss = None
#     _HAVE_FAISS = False

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    NearestNeighbors = None
    _HAVE_SKLEARN = False

# ---- Import your transformers-only embedding client (no sentence-transformers)
# Put embed_and_eval_stage1.py in the same folder as qwen_clients.py, or set PYTHONPATH accordingly.
from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig  # noqa: E402


# -------------------------
# small utils
# -------------------------

def _chunk_list(xs: List[Any], bs: int) -> Iterable[List[Any]]:
    """Yield list chunks of size bs."""
    bs = max(1, int(bs))
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


# -------------------------
# IO helpers
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# -------------------------
# Ontology parsing
# -------------------------

def _extract_id(term: Dict[str, Any]) -> str:
    return term.get("Id") or term.get("id") or term.get("hpo_id") or ""

def _extract_name(term: Dict[str, Any]) -> str:
    nm = term.get("Name") or term.get("name") or term.get("label") or ""
    if isinstance(nm, list):
        nm = nm[0] if nm else ""
    return str(nm).strip()

def _extract_parents(term: Dict[str, Any], max_n: int = 80) -> List[str]:
    """
    Your JSON may have:
      - Father: dict of parent_id -> true/1
      - Is_a: list or str
    """
    father = term.get("Father")
    parents: List[str] = []
    if isinstance(father, dict):
        parents = [k for k, v in father.items() if v]
    else:
        isa = term.get("Is_a") or term.get("is_a") or []
        if isinstance(isa, str):
            parents = [isa]
        elif isinstance(isa, list):
            parents = [str(x) for x in isa]
    parents = [p.strip() for p in parents if str(p).strip()]
    return parents[:max_n]

def build_ontology_maps(hpo_json: Any) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      id2name:    hpo_id -> name
      parents:    hpo_id -> [parent_ids]
      children:   hpo_id -> [child_ids]
    """
    if isinstance(hpo_json, dict):
        terms = [t for t in hpo_json.values() if isinstance(t, dict)]
    elif isinstance(hpo_json, list):
        terms = [t for t in hpo_json if isinstance(t, dict)]
    else:
        raise ValueError("Unsupported HPO JSON structure (must be dict or list).")

    id2name: Dict[str, str] = {}
    parents: Dict[str, List[str]] = {}

    for t in terms:
        hid = _extract_id(t)
        nm = _extract_name(t)
        if not hid or not nm:
            continue
        id2name[hid] = nm
        parents[hid] = _extract_parents(t)

    children: Dict[str, List[str]] = {hid: [] for hid in parents.keys()}
    for hid, ps in parents.items():
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(hid)

    return id2name, parents, children

def compute_ancestors(parents: Dict[str, List[str]], max_depth: int = 80) -> Dict[str, Set[str]]:
    """
    ancestors[hid] = set(all ancestors)
    (simple DFS with memoization)
    """
    memo: Dict[str, Set[str]] = {}

    def dfs(hid: str, depth: int) -> Set[str]:
        if hid in memo:
            return memo[hid]
        if depth >= max_depth:
            memo[hid] = set()
            return memo[hid]
        ps = parents.get(hid, [])
        out: Set[str] = set(ps)
        for p in ps:
            out |= dfs(p, depth + 1)
        memo[hid] = out
        return out

    for hid in parents.keys():
        dfs(hid, 0)

    return memo

def sibling_set(hid: str, parents: Dict[str, List[str]], children: Dict[str, List[str]]) -> Set[str]:
    """
    siblings = union of children of each parent, excluding self
    """
    sibs: Set[str] = set()
    for p in parents.get(hid, []):
        for c in children.get(p, []):
            if c != hid:
                sibs.add(c)
    return sibs


# -------------------------
# Stage1 item -> texts
# -------------------------

_WS = re.compile(r"\s+")

def clean_phrase(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s

def nonempty_list(xs: Any) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, str):
        xs = [xs]
    if not isinstance(xs, list):
        return [str(xs)]
    out: List[str] = []
    for x in xs:
        t = clean_phrase(str(x))
        if t:
            out.append(t)
    return out

def join_for_concat(name: str, s2: List[str], s3: List[str], s4: List[str], max_each: int = 6) -> str:
    """
    Single-string concat baseline: "name ; s2... ; s3... ; s4..."
    Keep it compact to avoid huge contexts.
    """
    parts: List[str] = []
    if name:
        parts.append(name)
    parts.extend(s2[:max_each])
    parts.extend(s3[:max_each])
    parts.extend(s4[:max_each])
    return clean_phrase(" ; ".join(parts))


# -------------------------
# Embedding builders (multi-view)
# -------------------------

@dataclass
class EmbedViews:
    E0_name: np.ndarray      # [N, D]
    E1_name_domain: np.ndarray
    E2_name_mech_domain: np.ndarray
    E3_concat: np.ndarray    # [N, D]

@dataclass
class DataPack:
    ids: List[str]
    names: List[str]
    s2: List[List[str]]
    s3: List[List[str]]
    s4: List[List[str]]

def build_datapack(stage1_items: List[Dict[str, Any]], limit: int = 0) -> DataPack:
    ids, names, s2s, s3s, s4s = [], [], [], [], []
    it = stage1_items[:limit] if limit and limit > 0 else stage1_items
    for obj in it:
        hid = obj.get("hpo_id") or ""
        nm = obj.get("name") or ""
        if not hid or not nm:
            continue
        ids.append(hid)
        names.append(clean_phrase(nm))
        s2s.append(nonempty_list(obj.get("scale_2_descriptive")))
        s3s.append(nonempty_list(obj.get("scale_3_mechanism")))
        s4s.append(nonempty_list(obj.get("scale_4_domain")))
    return DataPack(ids=ids, names=names, s2=s2s, s3=s3s, s4=s4s)

def mean_of_phrase_embs(
    client: Qwen3EmbeddingClient,
    phrases: List[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """
    phrases -> mean embedding vector (L2 normalized)
    """
    phrases = [clean_phrase(p) for p in (phrases or []) if clean_phrase(p)]
    if not phrases:
        # caller should avoid this; fallback to 1-dummy token to keep shape consistent
        phrases = ["[EMPTY]"]
    emb = client.encode(
        phrases,
        mode="raw",
        max_length=max_length,
        batch_size=batch_size,
        normalize=True,
        return_numpy=True,
    )  # [M,D]
    v = emb.mean(axis=0, keepdims=False)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)

def build_embeddings(
    client: Qwen3EmbeddingClient,
    dp: DataPack,
    max_length: int,
    batch_size: int,
) -> Tuple[EmbedViews, Dict[str, Any]]:
    """
    Build all 4 embedding variants.
    """
    N = len(dp.ids)
    if N == 0:
        raise RuntimeError("No valid items found in stage1 jsonl after filtering (hpo_id/name).")

    empty_counts = {"E0": 0, "E1": 0, "E2": 0, "E3": 0}

    # ---- E3 concat texts (one per item)
    concat_texts: List[str] = []
    for i in range(N):
        concat_texts.append(join_for_concat(dp.names[i], dp.s2[i], dp.s3[i], dp.s4[i]))

    # Batch embed concat
    pbar = tqdm(total=N, desc="EMB E3 concat", unit="term")
    E3_chunks: List[np.ndarray] = []
    for chunk in _chunk_list(concat_texts, batch_size):
        arr = client.encode(
            chunk,
            mode="raw",
            max_length=max_length,
            batch_size=batch_size,
            normalize=True,
            return_numpy=True,
        )
        E3_chunks.append(arr.astype(np.float32))
        pbar.update(len(chunk))
    pbar.close()
    E3 = np.concatenate(E3_chunks, axis=0)

    # ---- E0/E1/E2: per-item mean over phrase lists
    def build_view(view_name: str, make_phrases_fn):
        vecs: List[np.ndarray] = []
        p = tqdm(total=N, desc=f"EMB {view_name}", unit="term")
        for i in range(N):
            phrases = make_phrases_fn(i)
            phrases = [clean_phrase(x) for x in phrases if clean_phrase(x)]
            phrases = list(dict.fromkeys(phrases))  # de-dup keep order
            if not phrases:
                empty_counts[view_name] += 1
                phrases = [dp.names[i]]
            v = mean_of_phrase_embs(client, phrases, max_length=max_length, batch_size=batch_size)
            vecs.append(v)
            p.update(1)
        p.close()
        return np.stack(vecs, axis=0).astype(np.float32)

    E0 = build_view("E0", lambda i: [dp.names[i]])
    E1 = build_view("E1", lambda i: [dp.names[i]] + dp.s4[i])
    E2 = build_view("E2", lambda i: [dp.names[i]] + dp.s3[i] + dp.s4[i])

    stats = {
        "N": N,
        "empty_counts": empty_counts,
        "max_length": max_length,
        "batch_size": batch_size,
        "dim": int(E0.shape[1]),
    }
    return EmbedViews(E0_name=E0, E1_name_domain=E1, E2_name_mech_domain=E2, E3_concat=E3), stats


# -------------------------
# kNN retrieval
# -------------------------

def build_knn_index(vectors: np.ndarray, use_faiss: bool = True) -> Any:
    """
    vectors: [N, D] float32, already normalized
    """
    if use_faiss and _HAVE_FAISS:
        _, D = vectors.shape
        index = faiss.IndexFlatIP(D)
        index.add(vectors)
        return ("faiss", index)
    if _HAVE_SKLEARN:
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(vectors)
        return ("sklearn", nn)
    raise RuntimeError("Neither faiss nor sklearn is available for kNN. Install faiss (preferred).")

def knn_search(index_obj: Any, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    kind, idx = index_obj
    if kind == "faiss":
        S, I = idx.search(vectors, k)
        return I, S
    if kind == "sklearn":
        dist, I = idx.kneighbors(vectors, n_neighbors=k, return_distance=True)
        S = 1.0 - dist
        return I, S
    raise ValueError(f"Unknown index kind: {kind}")


# -------------------------
# Metrics
# -------------------------

def compute_metrics_for_view(
    view_name: str,
    ids: List[str],
    vectors: np.ndarray,
    parents: Dict[str, List[str]],
    children: Dict[str, List[str]],
    ancestors: Dict[str, Set[str]],
    ks: List[int],
    use_faiss: bool,
) -> Dict[str, Any]:
    N = len(ids)
    id2idx = {hid: i for i, hid in enumerate(ids)}

    sibs_local: List[Set[int]] = []
    for hid in ids:
        sib_ids = sibling_set(hid, parents, children)
        sib_idx = {id2idx[s] for s in sib_ids if s in id2idx}
        sibs_local.append(sib_idx)

    anc_local: List[Set[str]] = [ancestors.get(hid, set()) for hid in ids]

    maxK = max(ks)
    index = build_knn_index(vectors, use_faiss=use_faiss)
    I, S = knn_search(index, vectors, k=maxK + 1)

    neighs: List[List[int]] = []
    for i in range(N):
        row = I[i].tolist()
        out_idx = []
        for j in row:
            if j == i:
                continue
            out_idx.append(j)
            if len(out_idx) >= maxK:
                break
        neighs.append(out_idx)

    out: Dict[str, Any] = {"view": view_name, "N": N, "ks": ks, "metrics": {}}

    for K in ks:
        hit = 0
        sib_prec_sum = 0.0
        anc_jacc_sum = 0.0
        valid = 0

        for i in range(N):
            topk = neighs[i][:K]
            if not topk:
                continue
            valid += 1

            sib_set_i = sibs_local[i]
            if sib_set_i:
                n_sib = sum((j in sib_set_i) for j in topk)
                if n_sib > 0:
                    hit += 1
                sib_prec_sum += (n_sib / float(len(topk)))
            else:
                sib_prec_sum += 0.0

            ai = anc_local[i]
            if ai:
                jaccs = []
                for j in topk:
                    aj = anc_local[j]
                    if not aj:
                        jaccs.append(0.0)
                    else:
                        inter = len(ai & aj)
                        union = len(ai | aj)
                        jaccs.append(inter / union if union else 0.0)
                anc_jacc_sum += float(sum(jaccs) / len(jaccs))
            else:
                anc_jacc_sum += 0.0

        valid = max(1, valid)
        out["metrics"][f"@{K}"] = {
            "sibling_hit": hit / valid,
            "sibling_precision": sib_prec_sum / valid,
            "mean_ancestor_jaccard": anc_jacc_sum / valid,
        }

    return out


# -------------------------
# Report writer
# -------------------------

def markdown_report(
    run_cfg: Dict[str, Any],
    stats_embed: Dict[str, Any],
    stats_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    notes: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Stage1 Embedding A/B Evaluation Report\n")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")

    lines.append("## Run config\n```json")
    lines.append(json.dumps(run_cfg, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Data stats\n```json")
    lines.append(json.dumps(stats_data, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Embedding stats\n```json")
    lines.append(json.dumps(stats_embed, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Metrics summary\n")
    for r in results:
        view = r["view"]
        ks = r["ks"]
        lines.append(f"### {view}\n")
        lines.append("| K | sibling_hit | sibling_precision | mean_ancestor_jaccard |")
        lines.append("|---:|---:|---:|---:|")
        for K in ks:
            m = r["metrics"][f"@{K}"]
            lines.append(f"| {K} | {m['sibling_hit']:.4f} | {m['sibling_precision']:.4f} | {m['mean_ancestor_jaccard']:.4f} |")
        lines.append("")

    if notes:
        lines.append("## Notes")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("embed_and_eval_stage1")
    ap.add_argument(
        "--stage1_jsonl",
        type=str,
        default="/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/queries_refilled.jsonl",
        help="Stage1 refilled jsonl",
    )
    ap.add_argument(
        "--hpo_json",
        type=str,
        default="/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json",
        help="Ontology json (must contain Father/Is_a)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/out",
        help="Base output dir; script will create a new subdir with --tag",
    )
    ap.add_argument("--tag", type=str, default="", help="Subdir name; default=timestamp")

    ap.add_argument("--model_dir", type=str, default="/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B")
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / cuda:0 ... (empty=auto)")
    ap.add_argument("--dtype", type=str, default="", help="bf16/fp16/fp32 (empty=auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N items from jsonl")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--use_faiss", action="store_true", help="Use faiss if available (recommended)")
    args = ap.parse_args()

    tag = args.tag.strip() or now_tag()
    out_run_dir = os.path.join(args.out_dir, tag)
    ensure_dir(out_run_dir)

    report_path = os.path.join(out_run_dir, "report.md")
    metrics_path = os.path.join(out_run_dir, "metrics.json")

    run_cfg = {
        "stage1_jsonl": args.stage1_jsonl,
        "hpo_json": args.hpo_json,
        "out_run_dir": out_run_dir,
        "model_dir": args.model_dir,
        "device": args.device or "auto",
        "dtype": args.dtype or "auto",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "limit": args.limit,
        "ks": args.ks,
        "use_faiss": bool(args.use_faiss),
        "faiss_available": bool(_HAVE_FAISS),
        "sklearn_available": bool(_HAVE_SKLEARN),
    }

    notes: List[str] = []

    tqdm.write("[STEP 1/5] loading stage1 jsonl ...")
    stage1_items = read_jsonl(args.stage1_jsonl)
    stats_data: Dict[str, Any] = {
        "stage1_total_lines": len(stage1_items),
        "limit_applied": args.limit,
    }

    tqdm.write("[STEP 2/5] building datapack ...")
    dp = build_datapack(stage1_items, limit=args.limit)
    N = len(dp.ids)
    stats_data.update({
        "N_used": N,
        "example_first": {
            "hpo_id": dp.ids[0] if N else "",
            "name": dp.names[0] if N else "",
            "n_s2": len(dp.s2[0]) if N else 0,
            "n_s3": len(dp.s3[0]) if N else 0,
            "n_s4": len(dp.s4[0]) if N else 0,
        }
    })

    tqdm.write("[STEP 3/5] loading ontology + computing ancestors ...")
    with open(args.hpo_json, "r", encoding="utf-8") as f:
        hpo_json = json.load(f)

    _, parents, children = build_ontology_maps(hpo_json)
    ancestors = compute_ancestors(parents, max_depth=80)

    covered = sum(1 for hid in dp.ids if hid in parents)
    stats_data.update({
        "ontology_terms_total": len(parents),
        "ontology_coverage_in_eval_subset": f"{covered}/{N}",
    })
    if covered < N:
        notes.append("Some evaluated HPO IDs were not found in ontology parent map; their sibling/ancestor sets may be empty.")

    tqdm.write("[STEP 4/5] loading embedding model + building embeddings ...")
    emb_client = Qwen3EmbeddingClient(EmbeddingConfig(
        model_dir=args.model_dir,
        device=(args.device or None),
        dtype=(args.dtype or None),
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=True,
        attn_implementation=None,  # keep None unless flash_attn installed
        use_instruct=False,
    ))

    views, stats_embed = build_embeddings(
        client=emb_client,
        dp=dp,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    tqdm.write("[STEP 5/5] kNN retrieval + metrics ...")
    use_faiss = bool(args.use_faiss) and _HAVE_FAISS
    if bool(args.use_faiss) and not _HAVE_FAISS:
        notes.append("You requested --use_faiss but faiss is not available; falling back to sklearn brute-force (may be slow).")
    if (not use_faiss) and (not _HAVE_SKLEARN):
        raise RuntimeError("No retrieval backend: install faiss (preferred) or sklearn.")

    results: List[Dict[str, Any]] = []
    results.append(compute_metrics_for_view("E0_name", dp.ids, views.E0_name, parents, children, ancestors, args.ks, use_faiss=use_faiss))
    results.append(compute_metrics_for_view("E1_name_domain", dp.ids, views.E1_name_domain, parents, children, ancestors, args.ks, use_faiss=use_faiss))
    results.append(compute_metrics_for_view("E2_name_mech_domain", dp.ids, views.E2_name_mech_domain, parents, children, ancestors, args.ks, use_faiss=use_faiss))
    results.append(compute_metrics_for_view("E3_concat", dp.ids, views.E3_concat, parents, children, ancestors, args.ks, use_faiss=use_faiss))

    metrics_obj = {
        "run_cfg": run_cfg,
        "stats_data": stats_data,
        "stats_embed": stats_embed,
        "results": results,
        "notes": notes,
    }
    write_json(metrics_path, metrics_obj)

    md = markdown_report(
        run_cfg=run_cfg,
        stats_embed=stats_embed,
        stats_data=stats_data,
        results=results,
        notes=notes,
    )
    write_text(report_path, md)

    tqdm.write(f"[DONE] report: {report_path}")
    tqdm.write(f"[DONE] metrics: {metrics_path}")
    tqdm.write(f"[OUT]  dir   : {out_run_dir}")


if __name__ == "__main__":
    main()
