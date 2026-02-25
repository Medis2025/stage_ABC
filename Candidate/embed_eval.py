#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_and_eval_stage1.py  (FULL: eval + embedding-space "attention" validation + rich markdown)

What this script does
1) Load Stage1 refilled JSONL (queries_refilled.jsonl)
2) Load HPO ontology JSON (your enriched json with Father / Is_a)
3) Build 4 embedding variants (E0/E1/E2/E3) using Qwen3-Embedding-8B (transformers-only)
4) Build kNN index (FAISS preferred; sklearn fallback)
5) Compute ontology-structure metrics:
   - sibling_hit@K
   - sibling_precision@K
   - mean_ancestor_jaccard@K
6) Embedding-space "attention" validation (token contribution inspection)
   - For selected examples, we compute pooled embedding E and per-token cosine(H_t, E)
   - Print top contributing tokens and their scores
   - Also do simple ablation tests (remove top tokens / remove domain tokens) and show cosine drift
7) Write a rich Markdown report:
   - config + data stats + embedding stats
   - metrics tables for all views
   - examples: nearest neighbors, sibling/cousin comparisons
   - attention-inspection outputs + ablation outputs

Notes
- "Attention validated" here means: we validate what tokens dominate the pooled embedding direction
  (token->pooled cosine). This is NOT transformer attention weights. It's embedding-direction attribution,
  which is usually more actionable for debugging embedding behavior.

Run
  python embed_and_eval_stage1.py \
    --stage1_jsonl /cluster/home/gw/.../Candidate/queries_refilled.jsonl \
    --hpo_json     /cluster/home/gw/.../hpo_enriched_with_llm.json \
    --out_dir      /cluster/home/gw/.../Candidate/Candidate/out \
    --tag          stage1_eval_full \
    --max_length   512 \
    --batch_size   32 \
    --ks           10 50 100 \
    --use_faiss \
    --inspect_n    8 \
    --inspect_seed 13

"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set, Iterable

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# ---- Try FAISS (fast); fallback to sklearn (slow)
try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    faiss = None
    _HAVE_FAISS = False

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    NearestNeighbors = None
    _HAVE_SKLEARN = False

# ---- Import your transformers-only embedding client (no sentence-transformers)
# Requires: qwen_clients.py in same dir or in PYTHONPATH
from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig  # noqa: E402


# =============================================================================
# small utils
# =============================================================================

def _chunk_list(xs: List[Any], bs: int) -> Iterable[List[Any]]:
    """Yield list chunks of size bs."""
    bs = max(1, int(bs))
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


# =============================================================================
# IO helpers
# =============================================================================

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


# =============================================================================
# Ontology parsing
# =============================================================================

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
    DFS with memoization
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
    """siblings = union of children of each parent, excluding self"""
    sibs: Set[str] = set()
    for p in parents.get(hid, []):
        for c in children.get(p, []):
            if c != hid:
                sibs.add(c)
    return sibs


# =============================================================================
# Stage1 item -> texts
# =============================================================================

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


# =============================================================================
# Embedding builders (multi-view)
# =============================================================================

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


# =============================================================================
# kNN retrieval
# =============================================================================

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


# =============================================================================
# Metrics
# =============================================================================

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
    I, _S = knn_search(index, vectors, k=maxK + 1)

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


# =============================================================================
# Example mining (neighbors + attention validation)
# =============================================================================

def _safe_get_name(id2name: Dict[str, str], hid: str) -> str:
    return id2name.get(hid, hid)

def get_topk_neighbors(
    ids: List[str],
    vectors: np.ndarray,
    k: int,
    use_faiss: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      neigh_idx: [N,k] indices (self removed)
      neigh_sim: [N,k] cosine similarities
    """
    N = len(ids)
    index = build_knn_index(vectors, use_faiss=use_faiss)
    I, S = knn_search(index, vectors, k=k + 1)

    neigh_idx = np.zeros((N, k), dtype=np.int64)
    neigh_sim = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        row_i = I[i].tolist()
        row_s = S[i].tolist()
        out_i, out_s = [], []
        for j, sc in zip(row_i, row_s):
            if j == i:
                continue
            out_i.append(j)
            out_s.append(float(sc))
            if len(out_i) >= k:
                break
        if len(out_i) < k:
            # pad with self if needed (rare)
            out_i += [i] * (k - len(out_i))
            out_s += [1.0] * (k - len(out_s))
        neigh_idx[i, :] = np.array(out_i, dtype=np.int64)
        neigh_sim[i, :] = np.array(out_s, dtype=np.float32)

    return neigh_idx, neigh_sim

def sample_inspect_ids(
    ids: List[str],
    parents: Dict[str, List[str]],
    children: Dict[str, List[str]],
    n: int,
    seed: int,
) -> List[str]:
    """
    Sample a mix of:
      - nodes with many siblings
      - nodes with few/no siblings (within subset)
    """
    rng = random.Random(seed)
    sib_counts = []
    for hid in ids:
        sibs = sibling_set(hid, parents, children)
        sib_counts.append((hid, len(sibs)))
    sib_counts.sort(key=lambda x: x[1], reverse=True)

    top = [hid for hid, _ in sib_counts[:max(1, n // 2)]]
    tail = [hid for hid, _ in sib_counts[-max(1, n // 2):]]
    picked = list(dict.fromkeys(top + tail))  # unique preserve order
    if len(picked) < n:
        remain = [hid for hid in ids if hid not in set(picked)]
        rng.shuffle(remain)
        picked.extend(remain[: (n - len(picked))])
    return picked[:n]


# =============================================================================
# Token contribution inspection ("embedding-space attention")
# =============================================================================

def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Same logic as Qwen's recommended example:
    left-padding requires taking last token; otherwise take last non-pad token.
    """
    # attention_mask: [B,T]
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1, :]  # [B,H]
    seq_lens = attention_mask.sum(dim=1) - 1
    bsz = last_hidden.shape[0]
    return last_hidden[torch.arange(bsz, device=last_hidden.device), seq_lens, :]

@torch.no_grad()
def token_contrib_report(
    client: Qwen3EmbeddingClient,
    text: str,
    max_length: int,
    topn: int = 12,
) -> Dict[str, Any]:
    """
    Compute pooled embedding E, then per-token cosine(H_t, E).
    Returns a dict suitable for Markdown.
    """
    tok = client.tokenizer
    model = client.model
    device = next(model.parameters()).device

    enc = tok(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, return_dict=True)
    H = out.last_hidden_state[0]          # [T,H]
    mask = enc["attention_mask"][0]       # [T]

    # pooled
    E = _last_token_pool(out.last_hidden_state, enc["attention_mask"])[0]  # [H]
    E = F.normalize(E.float(), p=2, dim=-1)

    # per-token cosine
    Hn = F.normalize(H.float(), p=2, dim=-1)  # [T,H]
    cos = (Hn @ E.unsqueeze(-1)).squeeze(-1)  # [T]
    cos = cos.detach().cpu().numpy()

    ids = enc["input_ids"][0].detach().cpu().tolist()
    tokens = tok.convert_ids_to_tokens(ids)

    rows = []
    for t, c, m in zip(tokens, cos.tolist(), mask.detach().cpu().tolist()):
        if m == 0:
            continue
        rows.append((t, float(c)))

    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    top_rows = rows_sorted[:topn]

    return {
        "text": text,
        "top_tokens": [{"token": t, "cos_to_pooled": c} for t, c in top_rows],
        "all_len": int(sum(mask.detach().cpu().tolist())),
    }

def simple_ablation(
    client: Qwen3EmbeddingClient,
    text: str,
    max_length: int,
    remove_tokens: List[str],
) -> Dict[str, Any]:
    """
    Remove occurrences of certain token strings at whitespace-level (coarse but useful),
    then compute cosine(original, ablated).
    """
    def coarse_remove(s: str, toks: List[str]) -> str:
        s2 = s
        for tk in toks:
            # remove both " tk " and "tk" forms (rough)
            s2 = re.sub(rf"\b{re.escape(tk)}\b", "", s2, flags=re.IGNORECASE)
        s2 = clean_phrase(s2)
        return s2

    text2 = coarse_remove(text, remove_tokens)
    v1 = client.encode([text], mode="raw", max_length=max_length, batch_size=1, normalize=True, return_numpy=True)[0]
    v2 = client.encode([text2], mode="raw", max_length=max_length, batch_size=1, normalize=True, return_numpy=True)[0]
    cos = float(np.dot(v1, v2))
    return {
        "orig": text,
        "ablated": text2,
        "cosine(orig, ablated)": cos,
        "removed": remove_tokens,
    }


# =============================================================================
# Markdown report writer (rich)
# =============================================================================

def _md_table_metrics(view: Dict[str, Any]) -> str:
    view_name = view["view"]
    ks = view["ks"]
    lines = [f"### {view_name}\n"]
    lines.append("| K | sibling_hit | sibling_precision | mean_ancestor_jaccard |")
    lines.append("|---:|---:|---:|---:|")
    for K in ks:
        m = view["metrics"][f"@{K}"]
        lines.append(
            f"| {K} | {m['sibling_hit']:.4f} | {m['sibling_precision']:.4f} | {m['mean_ancestor_jaccard']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)

def _md_neighbors_section(
    title: str,
    ids: List[str],
    id2name: Dict[str, str],
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
    inspect_ids: List[str],
    k: int,
) -> str:
    id2idx = {hid: i for i, hid in enumerate(ids)}
    lines = [f"## {title}\n"]
    for hid in inspect_ids:
        if hid not in id2idx:
            continue
        i = id2idx[hid]
        lines.append(f"### {hid} — {_safe_get_name(id2name, hid)}\n")
        lines.append("| rank | neighbor_id | neighbor_name | cosine |")
        lines.append("|---:|---:|---|---:|")
        for r in range(k):
            j = int(neigh_idx[i, r])
            sc = float(neigh_sim[i, r])
            nhid = ids[j]
            lines.append(f"| {r+1} | {nhid} | {_safe_get_name(id2name, nhid)} | {sc:.4f} |")
        lines.append("")
    return "\n".join(lines)

def _md_attention_section(attn_reports: List[Dict[str, Any]], ablations: List[Dict[str, Any]]) -> str:
    lines = ["## Embedding-space attention validation (token contribution)\n"]
    lines.append(
        "This section inspects which tokens dominate the pooled embedding direction. "
        "We compute cosine(token_hidden, pooled_embedding) for each token.\n"
    )

    for rep in attn_reports:
        lines.append(f"### Text: `{rep['text']}`\n")
        lines.append(f"- token_count: `{rep['all_len']}`\n")
        lines.append("| rank | token | cos_to_pooled |")
        lines.append("|---:|---|---:|")
        for i, row in enumerate(rep["top_tokens"], start=1):
            lines.append(f"| {i} | `{row['token']}` | {row['cos_to_pooled']:.4f} |")
        lines.append("")

    if ablations:
        lines.append("### Ablation sanity checks\n")
        lines.append(
            "We do coarse whitespace-level ablation to see how much the embedding changes when removing selected tokens.\n"
        )
        for ab in ablations:
            lines.append("```json")
            lines.append(json.dumps(ab, ensure_ascii=False, indent=2))
            lines.append("```\n")

    return "\n".join(lines)

def markdown_report(
    run_cfg: Dict[str, Any],
    stats_embed: Dict[str, Any],
    stats_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    notes: List[str],
    extra_sections: List[str],
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
        lines.append(_md_table_metrics(r))

    if notes:
        lines.append("## Notes")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    if extra_sections:
        lines.extend(extra_sections)

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

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

    # inspection
    ap.add_argument("--inspect_n", type=int, default=8, help="How many HPO items to print examples/attention for")
    ap.add_argument("--inspect_k", type=int, default=10, help="How many neighbors to show in examples")
    ap.add_argument("--inspect_seed", type=int, default=13, help="Random seed for inspection sampling")
    ap.add_argument("--inspect_view", type=str, default="E1", choices=["E0", "E1", "E2", "E3"],
                    help="Which embedding view to use for printing neighbors/attention examples")
    ap.add_argument("--attn_topn", type=int, default=12, help="Top token contributions to show")
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
        "inspect": {
            "inspect_n": args.inspect_n,
            "inspect_k": args.inspect_k,
            "inspect_seed": args.inspect_seed,
            "inspect_view": args.inspect_view,
            "attn_topn": args.attn_topn,
        },
    }

    notes: List[str] = []

    # STEP 1
    tqdm.write("[STEP 1/6] loading stage1 jsonl ...")
    stage1_items = read_jsonl(args.stage1_jsonl)
    stats_data: Dict[str, Any] = {
        "stage1_total_lines": len(stage1_items),
        "limit_applied": args.limit,
    }

    # STEP 2
    tqdm.write("[STEP 2/6] building datapack ...")
    dp = build_datapack(stage1_items, limit=args.limit)
    N = len(dp.ids)
    if N == 0:
        raise RuntimeError("No usable items from stage1 jsonl.")
    stats_data.update({
        "N_used": N,
        "example_first": {
            "hpo_id": dp.ids[0],
            "name": dp.names[0],
            "n_s2": len(dp.s2[0]),
            "n_s3": len(dp.s3[0]),
            "n_s4": len(dp.s4[0]),
        }
    })

    # STEP 3
    tqdm.write("[STEP 3/6] loading ontology + computing ancestors ...")
    with open(args.hpo_json, "r", encoding="utf-8") as f:
        hpo_json = json.load(f)

    id2name, parents, children = build_ontology_maps(hpo_json)
    ancestors = compute_ancestors(parents, max_depth=80)

    covered = sum(1 for hid in dp.ids if hid in parents)
    stats_data.update({
        "ontology_terms_total": len(parents),
        "ontology_coverage_in_eval_subset": f"{covered}/{N}",
    })
    if covered < N:
        notes.append("Some evaluated HPO IDs were not found in ontology parent map; their sibling/ancestor sets may be empty.")

    # STEP 4
    tqdm.write("[STEP 4/6] loading embedding model + building embeddings ...")
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

    # STEP 5
    tqdm.write("[STEP 5/6] kNN retrieval + metrics ...")
    use_faiss = bool(args.use_faiss) and _HAVE_FAISS
    backend = "faiss" if use_faiss else ("sklearn" if _HAVE_SKLEARN else "none")
    stats_data["retrieval_backend"] = backend

    if bool(args.use_faiss) and not _HAVE_FAISS:
        notes.append("You requested --use_faiss but faiss is not available; falling back to sklearn brute-force (may be slow).")
    if (not use_faiss) and (not _HAVE_SKLEARN):
        raise RuntimeError("No retrieval backend: install faiss (preferred) or sklearn.")

    results: List[Dict[str, Any]] = []
    results.append(compute_metrics_for_view("E0_name", dp.ids, views.E0_name, parents, children, ancestors, args.ks, use_faiss=use_faiss))
    results.append(compute_metrics_for_view("E1_name_domain", dp.ids, views.E1_name_domain, parents, children, ancestors, args.ks, use_faiss=use_faiss))
    results.append(compute_metrics_for_view("E2_name_mech_domain", dp.ids, views.E2_name_mech_domain, parents, children, ancestors, args.ks, use_faiss=use_faiss))
    results.append(compute_metrics_for_view("E3_concat", dp.ids, views.E3_concat, parents, children, ancestors, args.ks, use_faiss=use_faiss))

    # STEP 6
    tqdm.write("[STEP 6/6] examples + embedding-space attention inspection ...")

    # pick which view for example printing
    if args.inspect_view == "E0":
        v = views.E0_name
        view_title = "E0_name"
    elif args.inspect_view == "E1":
        v = views.E1_name_domain
        view_title = "E1_name_domain"
    elif args.inspect_view == "E2":
        v = views.E2_name_mech_domain
        view_title = "E2_name_mech_domain"
    else:
        v = views.E3_concat
        view_title = "E3_concat"

    inspect_ids = sample_inspect_ids(dp.ids, parents, children, n=args.inspect_n, seed=args.inspect_seed)

    neigh_idx, neigh_sim = get_topk_neighbors(dp.ids, v, k=args.inspect_k, use_faiss=use_faiss)

    # Build attention reports for selected examples (use E3 concat text as default "inspection text")
    # because it contains richer signals. Also include name-only as a contrast.
    attn_reports: List[Dict[str, Any]] = []
    ablations: List[Dict[str, Any]] = []

    id2idx = {hid: i for i, hid in enumerate(dp.ids)}
    for hid in tqdm(inspect_ids, desc="ATTN inspect", unit="term"):
        i = id2idx[hid]
        name = dp.names[i]
        concat = join_for_concat(name, dp.s2[i], dp.s3[i], dp.s4[i])

        # token contribution: name-only vs concat
        rep_name = token_contrib_report(emb_client, name, max_length=args.max_length, topn=args.attn_topn)
        rep_concat = token_contrib_report(emb_client, concat, max_length=args.max_length, topn=args.attn_topn)
        attn_reports.append({
            "text": f"{hid} | NAME: {name}",
            "top_tokens": rep_name["top_tokens"],
            "all_len": rep_name["all_len"],
        })
        attn_reports.append({
            "text": f"{hid} | CONCAT: {concat}",
            "top_tokens": rep_concat["top_tokens"],
            "all_len": rep_concat["all_len"],
        })

        # ablation: remove top-3 contributing *word-like* tokens from CONCAT (coarse)
        # We filter tokens like "Ġxxx"/"▁xxx" etc and strip common markers.
        top_tokens = [t["token"] for t in rep_concat["top_tokens"][:3]]
        cleaned = []
        for tk in top_tokens:
            tk2 = tk
            tk2 = tk2.replace("▁", "").replace("Ġ", "")
            tk2 = re.sub(r"^[^\w]+|[^\w]+$", "", tk2)
            if tk2 and len(tk2) >= 3:
                cleaned.append(tk2)
        cleaned = list(dict.fromkeys(cleaned))[:3]
        if cleaned:
            ablations.append(simple_ablation(emb_client, concat, max_length=args.max_length, remove_tokens=cleaned))

    extra_sections: List[str] = []
    extra_sections.append(_md_neighbors_section(
        title=f"Nearest neighbor examples (view={view_title}, backend={backend})",
        ids=dp.ids,
        id2name=id2name,
        neigh_idx=neigh_idx,
        neigh_sim=neigh_sim,
        inspect_ids=inspect_ids,
        k=args.inspect_k,
    ))
    extra_sections.append(_md_attention_section(attn_reports, ablations))

    # Write outputs
    metrics_obj = {
        "run_cfg": run_cfg,
        "stats_data": stats_data,
        "stats_embed": stats_embed,
        "results": results,
        "notes": notes,
        "examples": {
            "inspect_view": view_title,
            "inspect_ids": inspect_ids,
            "neighbors_k": args.inspect_k,
            "backend": backend,
        },
        "attention_validation": {
            "method": "cosine(token_hidden, pooled_embedding) using last_token_pool",
            "inspect_n": args.inspect_n,
            "attn_topn": args.attn_topn,
        }
    }
    write_json(metrics_path, metrics_obj)

    md = markdown_report(
        run_cfg=run_cfg,
        stats_embed=stats_embed,
        stats_data=stats_data,
        results=results,
        notes=notes,
        extra_sections=extra_sections,
    )
    write_text(report_path, md)

    tqdm.write(f"[DONE] report: {report_path}")
    tqdm.write(f"[DONE] metrics: {metrics_path}")
    tqdm.write(f"[OUT]  dir   : {out_run_dir}")


if __name__ == "__main__":
    main()
