#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stageB2_recluster_full_local_qwen.py

Full-process B2 reclustering over ALL HPOs:
- Load master HPO ids + names from B2 out/Def_embed
- Load Stage2 Domain embeddings (E1_name_domain) aligned to master order
- Load/build Def embeddings (E_def.npy preferred)
- (Optional) build Med / Weak prototypes with STRICT HPO-level masks (never ignore)
- Fuse vectors V
- Round-1 kNN: HNSW provides idx ONLY
- Recompute edge weights EXACT by dot: w(i,j)=float(V[i] @ V[j])
- Leiden clustering
- Cluster-aware smoothing (eta)
- Round-2 kNN (idx only) + exact dot weights
- Round-2 Leiden
- Save ALL outputs to an outdir: vectors, idx, sims, labels, graphs meta, summaries, jsonl clusters

IMPORTANT:
- kNN similarities used in graph are ALWAYS exact dot computed from V (or V2) and idx,
  so the graph weights match exact cosine (since V is L2-normalized).
- This script processes the entire set of HPO ids, not a single target test.

Deps:
  pip install numpy tqdm hnswlib igraph leidenalg

Run example:
  python3 stageB2_recluster_full_local_qwen.py \
    --b2_out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/out \
    --stage2_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE2/cluster_data/frozen_E1K10_20260119_160424 \
    --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/STAGE3/phaseB/B1B2B3/B2_embed/recluster_out/run_20260204 \
    --K 50 --resolution 1.0 --eta 0.08 \
    --use_med --use_weak

What it saves (under --out_dir):
  - meta.json
  - summary.json
  - master_hpo_ids.json
  - E_dom.npy / E_def.npy / V0_fused.npy / V1_smooth.npy
  - knn_round1_idx.npy / knn_round1_sim_exact.npy
  - knn_round2_idx.npy / knn_round2_sim_exact.npy
  - labels_round1.npy / labels_round2.npy
  - clusters_round1.jsonl / clusters_round2.jsonl
  - cluster_sizes_round1.json / cluster_sizes_round2.json
  - prototypes_med.npy / counts_med.npy   (if use_med)
  - prototypes_weak.npy / counts_weak.npy (if use_weak)
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import hnswlib
import igraph as ig
import leidenalg
from tqdm import tqdm

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


# =============================================================================
# IO utils
# =============================================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(p: str, obj: Any) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(p: str, rows: List[Dict[str, Any]]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _ensure_file(p: str, what: str) -> None:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{what} not found: {p}")

def _ensure_dir(p: str, what: str) -> None:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{what} not found: {p}")

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# =============================================================================
# math utils
# =============================================================================

def l2_normalize_mat(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / (np.linalg.norm(x) + 1e-12)

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


# =============================================================================
# path resolve
# =============================================================================

def resolve_b2_out_dir(b2_out_dir: str) -> str:
    """
    Accept either:
      - .../B2_embed/out
      - .../B2_embed  (auto append /out)
    Detect by presence of Def_embed.
    """
    b2_out_dir = os.path.abspath(b2_out_dir)
    if os.path.isdir(os.path.join(b2_out_dir, "Def_embed")):
        return b2_out_dir
    cand = os.path.join(b2_out_dir, "out")
    if os.path.isdir(os.path.join(cand, "Def_embed")):
        return cand
    raise FileNotFoundError(
        "Cannot locate B2 out dir. Expected Def_embed under either:\n"
        f"  {b2_out_dir}/Def_embed\n"
        f"  {b2_out_dir}/out/Def_embed"
    )


# =============================================================================
# master ids + names
# =============================================================================

def load_hpo_ids_fallback_jsonl(p: str) -> List[str]:
    ids: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                if line.startswith("HP:"):
                    ids.append(line)
                continue
            if isinstance(obj, str) and obj.startswith("HP:"):
                ids.append(obj)
            elif isinstance(obj, dict):
                hid = obj.get("hpo_id") or obj.get("id")
                if isinstance(hid, str) and hid.startswith("HP:"):
                    ids.append(hid)
    return ids

def load_master_hpo_ids(def_embed_dir: str) -> List[str]:
    p = os.path.join(def_embed_dir, "hpo_ids.json")
    _ensure_file(p, "Def_embed/hpo_ids.json")

    try:
        obj = read_json(p)
    except Exception:
        obj = None

    if isinstance(obj, list):
        ids = [x for x in obj if isinstance(x, str) and x.startswith("HP:")]
        if ids:
            return ids
        raise ValueError(f"hpo_ids.json list contains no valid HP: ids: {p}")

    if isinstance(obj, dict):
        for key in ("hpo_ids", "ids", "HPO_IDS"):
            if key in obj and isinstance(obj[key], list):
                ids = [x for x in obj[key] if isinstance(x, str) and x.startswith("HP:")]
                if ids:
                    return ids
        keys = [k for k in obj.keys() if isinstance(k, str) and k.startswith("HP:")]
        if keys:
            return sorted(keys)

    ids2 = load_hpo_ids_fallback_jsonl(p)
    if ids2:
        return ids2

    raise ValueError(f"hpo_ids.json is empty or unrecognized format: {p}")

def load_hpo_names(def_source_jsonl: str) -> Dict[str, str]:
    h2n: Dict[str, str] = {}
    for obj in iter_jsonl(def_source_jsonl):
        hid = obj.get("hpo_id")
        name = obj.get("name", "")
        if hid:
            h2n[hid] = name
    return h2n


# =============================================================================
# Stage2 Domain embedding aligned
# =============================================================================

def load_stage2_ids(stage2_ids_json: str) -> List[str]:
    obj = read_json(stage2_ids_json)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, str)]
    if isinstance(obj, dict):
        if "ids" in obj and isinstance(obj["ids"], list):
            return [x for x in obj["ids"] if isinstance(x, str)]
        if "hpo_ids" in obj and isinstance(obj["hpo_ids"], list):
            return [x for x in obj["hpo_ids"] if isinstance(x, str)]
    raise ValueError(f"Unrecognized ids.json format: {stage2_ids_json}")

def load_domain_aligned(stage2_dir: str, master_hpo_ids: List[str]) -> np.ndarray:
    emb_dir = os.path.join(stage2_dir, "layer1_embedding_E1_name_domain")
    _ensure_dir(emb_dir, "Stage2 layer1_embedding_E1_name_domain")

    pE = os.path.join(emb_dir, "E1_name_domain.npy")
    pids = os.path.join(emb_dir, "ids.json")
    _ensure_file(pE, "E1_name_domain.npy")
    _ensure_file(pids, "Stage2 ids.json")

    E = np.load(pE).astype(np.float32, copy=False)  # [N2,D]
    ids = load_stage2_ids(pids)                     # [N2]
    if E.shape[0] != len(ids):
        raise ValueError(f"Stage2 mismatch: E rows={E.shape[0]} but ids={len(ids)}")

    pos = {h: i for i, h in enumerate(ids)}
    missing = [h for h in master_hpo_ids if h not in pos]
    if missing:
        raise ValueError(f"Domain ids.json missing {len(missing)} HPOs; example: {missing[:5]}")

    out = np.zeros((len(master_hpo_ids), E.shape[1]), dtype=np.float32)
    for i, h in enumerate(master_hpo_ids):
        out[i] = E[pos[h]]

    return l2_normalize_mat(out)


# =============================================================================
# Def embedding
# =============================================================================

def build_def_embeddings(
    def_embed_dir: str,
    master_hpo_ids: List[str],
    hpo2name: Dict[str, str],
    *,
    qwen_client: Qwen3EmbeddingClient,
    qwen_mode: str = "doc",
    force_reencode: bool = False,
) -> Tuple[np.ndarray, str]:
    pE = os.path.join(def_embed_dir, "E_def.npy")
    if (not force_reencode) and os.path.isfile(pE):
        E = np.load(pE).astype(np.float32, copy=False)
        if E.shape[0] == len(master_hpo_ids):
            return l2_normalize_mat(E), "loaded E_def.npy"

    def _make_text(hpo_id: str) -> str:
        name = hpo2name.get(hpo_id, "")
        return f"{name} ({hpo_id})" if name else hpo_id

    texts = [_make_text(h) for h in master_hpo_ids]
    E = qwen_client.encode(texts, mode=qwen_mode, return_numpy=True)
    E = np.asarray(E, dtype=np.float32)
    return l2_normalize_mat(E), "qwen-encoded(name+hpo_id)"


# =============================================================================
# Med/Weak prototypes (STRICT HPO-level mask)
# =============================================================================

def load_index_ranges(p_index_json: str) -> Dict[str, Tuple[int, int]]:
    idx = read_json(p_index_json)
    out: Dict[str, Tuple[int, int]] = {}
    for k, v in idx.items():
        if isinstance(v, list) and len(v) == 2:
            out[k] = (int(v[0]), int(v[1]))
        elif isinstance(v, dict) and "s" in v and "e" in v:
            out[k] = (int(v["s"]), int(v["e"]))
    return out

def topk_mean_pool(rows: np.ndarray, k: int) -> np.ndarray:
    if rows.shape[0] == 1:
        return rows[0]
    c = rows.mean(axis=0)
    c = l2_normalize_vec(c)
    scores = rows @ c
    kk = min(k, rows.shape[0])
    idx = topk_indices(scores, kk)
    v = rows[idx].mean(axis=0)
    return l2_normalize_vec(v)

def build_pool_prototypes_strict_mask(
    *,
    pool_name: str,
    master_hpo_ids: List[str],
    E_pool: np.ndarray,
    mask_hpo: np.ndarray,            # shape [N_master]
    ranges: Dict[str, Tuple[int, int]],
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    STRICT:
      - mask_hpo is HPO-level (len == N_master)
      - never ignore; if mismatch -> raise
      - if mask_hpo[i]==False => prototype zero
      - if mask_hpo[i]==True => must have a valid range within E_pool bounds else error
    """
    E_pool = l2_normalize_mat(E_pool)
    mask_hpo = np.asarray(mask_hpo).astype(np.bool_, copy=False)

    N = len(master_hpo_ids)
    if mask_hpo.ndim != 1 or mask_hpo.shape[0] != N:
        raise ValueError(
            f"{pool_name} mask must be HPO-level 1D length N_master={N}, got shape={mask_hpo.shape}"
        )

    D = E_pool.shape[1]
    P = np.zeros((N, D), dtype=np.float32)
    n = np.zeros((N,), dtype=np.int32)

    # debug prints (requested)
    print(f"[DEBUG {pool_name}] E_pool.shape    =", E_pool.shape)
    print(f"[DEBUG {pool_name}] mask.shape      =", mask_hpo.shape)
    print(f"[DEBUG {pool_name}] mask.ndim       =", mask_hpo.ndim)
    print(f"[DEBUG {pool_name}] ranges size     =", len(ranges))
    some = next(iter(ranges.items()))
    print(f"[DEBUG {pool_name}] example range   =", some)

    # stats
    have_mask = int(mask_hpo.sum())
    have_range = 0
    used = 0
    empty_range = 0

    pbar = tqdm(total=N, desc=f"PROTO {pool_name}", unit="hpo")
    for i, h in enumerate(master_hpo_ids):
        pbar.update(1)
        if not bool(mask_hpo[i]):
            continue

        if h not in ranges:
            # strict: mask says yes but no range => treat as empty evidence (count=0), do not crash
            # (If you prefer hard fail, change to raise)
            empty_range += 1
            continue

        have_range += 1
        s, e = ranges[h]
        if e <= s:
            empty_range += 1
            continue

        if s < 0 or e > E_pool.shape[0]:
            raise ValueError(
                f"{pool_name} range out of bounds for {h}: (s,e)=({s},{e}) but E_pool rows={E_pool.shape[0]}"
            )

        rows = E_pool[s:e]
        if rows.shape[0] == 0:
            empty_range += 1
            continue

        n[i] = rows.shape[0]
        P[i] = topk_mean_pool(rows, topk)
        used += 1

    pbar.close()

    stats = {
        "N_master": N,
        "pool_rows": int(E_pool.shape[0]),
        "mask_true": have_mask,
        "mask_true_with_range": have_range,
        "prototypes_used": used,
        "empty_or_missing_range_under_mask": empty_range,
    }
    return P, n, stats


# =============================================================================
# Fusion
# =============================================================================

def fuse_vectors(
    E_def: np.ndarray,
    E_dom: np.ndarray,
    *,
    lam: float = 0.6,
    use_med: bool = False,
    P_med: Optional[np.ndarray] = None,
    n_med: Optional[np.ndarray] = None,
    K_med_sat: int = 10,
    alpha: float = 0.15,
    use_weak: bool = False,
    P_weak: Optional[np.ndarray] = None,
    n_weak: Optional[np.ndarray] = None,
    K_weak_sat: int = 20,
    beta: float = 0.05,
) -> np.ndarray:
    if E_def.shape != E_dom.shape:
        raise ValueError("E_def and E_dom must have same shape")

    N, _ = E_def.shape
    E_def = l2_normalize_mat(E_def)
    E_dom = l2_normalize_mat(E_dom)

    A = l2_normalize_mat(lam * E_def + (1.0 - lam) * E_dom)
    V = A.copy()

    if use_med:
        if P_med is None or n_med is None:
            raise ValueError("use_med=True but P_med/n_med not provided")
        P_med = l2_normalize_mat(P_med)
        delta = np.maximum(0.0, np.sum(A * P_med, axis=1))
        denom = math.log(1.0 + K_med_sat)
        g = np.minimum(1.0, np.log1p(n_med.astype(np.float32)) / (denom + 1e-12))
        V = V + (alpha * (g * delta)).reshape(N, 1) * P_med

    if use_weak:
        if P_weak is None or n_weak is None:
            raise ValueError("use_weak=True but P_weak/n_weak not provided")
        P_weak = l2_normalize_mat(P_weak)
        delta = np.maximum(0.0, np.sum(A * P_weak, axis=1))
        denom = math.log(1.0 + K_weak_sat)
        g = np.minimum(1.0, np.log1p(n_weak.astype(np.float32)) / (denom + 1e-12))
        V = V + (beta * (g * delta)).reshape(N, 1) * P_weak

    return l2_normalize_mat(V)


# =============================================================================
# kNN: HNSW idx only + exact dot weights
# =============================================================================

def hnsw_knn_idx_only(
    V: np.ndarray,
    *,
    K: int,
    M: int,
    efC: int,
    efS: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Return idx [N,K] from HNSW. No similarity trusted.
    """
    V = V.astype(np.float32, copy=False)
    N, D = V.shape
    index = hnswlib.Index(space="ip", dim=D)
    index.set_num_threads(0)  # auto
    index.init_index(max_elements=N, ef_construction=efC, M=M, random_seed=seed)
    index.add_items(V, np.arange(N))
    index.set_ef(efS)
    idx, _ = index.knn_query(V, k=K + 1)
    return idx[:, 1:].astype(np.int32)

def recompute_knn_sim_from_idx(
    V: np.ndarray,
    idx: np.ndarray,
    *,
    desc: str = "recompute sim",
) -> np.ndarray:
    """
    sim[i,t] = V[i] @ V[idx[i,t]]  (exact dot)
    """
    V = V.astype(np.float32, copy=False)
    N, K = idx.shape
    sim = np.empty((N, K), dtype=np.float32)
    pbar = tqdm(total=N, desc=desc, unit="hpo")
    for i in range(N):
        js = idx[i]
        sim[i] = V[js] @ V[i]
        pbar.update(1)
    pbar.close()
    return sim


# =============================================================================
# Graph + Leiden
# =============================================================================

def knn_to_igraph(idx: np.ndarray, sim: np.ndarray, *, sym: str = "max", sim_min: float = -1.0) -> ig.Graph:
    """
    Build undirected weighted graph from kNN lists.
    sym:
      - "max": w(i,j)=max(w_ij, w_ji)
      - "mean": average if both exist else existing
    """
    N, K = idx.shape
    wdir: Dict[Tuple[int, int], float] = {}

    pbar = tqdm(total=N, desc="GRAPH collect directed weights", unit="hpo")
    for i in range(N):
        js = idx[i]
        ws = sim[i]
        for j, w in zip(js.tolist(), ws.tolist()):
            if i == int(j):
                continue
            if float(w) < sim_min:
                continue
            wdir[(i, int(j))] = float(w)
        pbar.update(1)
    pbar.close()

    edges = []
    weights = []
    seen = set()

    pbar2 = tqdm(total=len(wdir), desc="GRAPH symmetrize", unit="edge")
    for (i, j), wij in wdir.items():
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            pbar2.update(1)
            continue
        wji = wdir.get((j, i), None)
        if wji is None:
            w = wij
        else:
            if sym == "mean":
                w = 0.5 * (wij + float(wji))
            else:
                w = max(wij, float(wji))
        seen.add((a, b))
        edges.append((a, b))
        weights.append(float(w))
        pbar2.update(1)
    pbar2.close()

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    return g

def leiden_cluster(g: ig.Graph, *, resolution: float, seed: int) -> np.ndarray:
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.array(part.membership, dtype=np.int32)


# =============================================================================
# Reclustering smoothing
# =============================================================================

def cluster_smooth(V: np.ndarray, labels: np.ndarray, *, eta: float) -> np.ndarray:
    V = V.astype(np.float32, copy=False)
    labels = labels.astype(np.int32, copy=False)
    N, D = V.shape
    C = int(labels.max()) + 1

    mu = np.zeros((C, D), dtype=np.float32)
    cnt = np.zeros((C,), dtype=np.int32)

    pbar = tqdm(total=N, desc="SMOOTH accumulate", unit="hpo")
    for i in range(N):
        c = int(labels[i])
        mu[c] += V[i]
        cnt[c] += 1
        pbar.update(1)
    pbar.close()

    for c in range(C):
        if cnt[c] > 0:
            mu[c] = l2_normalize_vec(mu[c] / float(cnt[c]))
        else:
            mu[c] = 0.0

    V2 = (1.0 - eta) * V + eta * mu[labels]
    return l2_normalize_mat(V2)


# =============================================================================
# Save cluster jsonl
# =============================================================================

def cluster_sizes(labels: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    labels = labels.astype(np.int32)
    for c in np.unique(labels):
        out[str(int(c))] = int((labels == c).sum())
    return out

def write_clusters_jsonl(
    out_path: str,
    master_hpo_ids: List[str],
    hpo2name: Dict[str, str],
    labels: np.ndarray,
) -> None:
    rows: List[Dict[str, Any]] = []
    labels = labels.astype(np.int32)
    pbar = tqdm(total=len(master_hpo_ids), desc=f"WRITE {os.path.basename(out_path)}", unit="hpo")
    for i, hid in enumerate(master_hpo_ids):
        rows.append({
            "hpo_id": hid,
            "name": hpo2name.get(hid, ""),
            "cluster": int(labels[i]),
        })
        pbar.update(1)
    pbar.close()
    write_jsonl(out_path, rows)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser("stageB2_recluster_full_local_qwen")

    ap.add_argument("--b2_out_dir", required=True, help=".../B2_embed/out OR .../B2_embed (auto out/)")
    ap.add_argument("--stage2_dir", required=True, help=".../STAGE2/cluster_data/frozen_E1K10_...")

    ap.add_argument("--out_dir", required=True, help="Output directory for all saved npy/json/jsonl")

    # Qwen
    ap.add_argument("--qwen_model_dir", default="/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B")
    ap.add_argument("--qwen_bs", type=int, default=32)
    ap.add_argument("--qwen_maxlen", type=int, default=512)
    ap.add_argument("--qwen_dtype", default=None)
    ap.add_argument("--qwen_attn_impl", default=None)
    ap.add_argument("--force_def_reencode", action="store_true", help="Force Qwen re-encode def even if E_def.npy exists")

    # Fusion
    ap.add_argument("--lam", type=float, default=0.6)

    ap.add_argument("--use_med", action="store_true")
    ap.add_argument("--use_weak", action="store_true")

    ap.add_argument("--med_topk", type=int, default=5)
    ap.add_argument("--weak_topk", type=int, default=5)

    ap.add_argument("--K_med_sat", type=int, default=10)
    ap.add_argument("--K_weak_sat", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.05)

    # kNN
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--efS", type=int, default=200)
    ap.add_argument("--sim_min", type=float, default=-1.0)

    # Leiden
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Smooth
    ap.add_argument("--eta", type=float, default=0.08)

    args = ap.parse_args()

    # resolve + ensure
    _ensure_dir(args.b2_out_dir, "b2_out_dir (as provided)")
    _ensure_dir(args.stage2_dir, "stage2_dir")
    args.b2_out_dir = resolve_b2_out_dir(args.b2_out_dir)

    ensure_dir(args.out_dir)

    def_embed_dir = os.path.join(args.b2_out_dir, "Def_embed")
    med_embed_dir = os.path.join(args.b2_out_dir, "Medium_embed")
    weak_embed_dir = os.path.join(args.b2_out_dir, "weak")

    _ensure_dir(def_embed_dir, "Def_embed")
    def_source = os.path.join(def_embed_dir, "def_source.jsonl")
    _ensure_file(def_source, "def_source.jsonl")

    print("[PATH] resolved b2_out_dir =", args.b2_out_dir)
    print("[PATH] def_embed_dir      =", def_embed_dir)
    print("[PATH] stage2_dir         =", args.stage2_dir)
    print("[PATH] out_dir            =", os.path.abspath(args.out_dir))

    meta = {
        "created_at": now_ts(),
        "b2_out_dir": args.b2_out_dir,
        "stage2_dir": args.stage2_dir,
        "out_dir": os.path.abspath(args.out_dir),
        "params": vars(args),
        "notes": [
            "kNN backend is HNSW for idx only; graph weights are recomputed by exact dot(V[i],V[j]).",
            "All vectors are L2-normalized => dot == cosine similarity.",
        ],
    }
    write_json(os.path.join(args.out_dir, "meta.json"), meta)

    # master ids + names
    print("[1] Load master hpo ids + names ...")
    master_hpo_ids = load_master_hpo_ids(def_embed_dir)
    hpo2name = load_hpo_names(def_source)
    N = len(master_hpo_ids)
    print(f"    N_master = {N}")
    write_json(os.path.join(args.out_dir, "master_hpo_ids.json"), master_hpo_ids)

    # Qwen client
    print("[2] Init Qwen3EmbeddingClient ...")
    qcfg = EmbeddingConfig(
        model_dir=args.qwen_model_dir,
        batch_size=args.qwen_bs,
        max_length=args.qwen_maxlen,
        dtype=args.qwen_dtype,
        attn_implementation=args.qwen_attn_impl,
        use_instruct=False,
        normalize=True,
    )
    qwen = Qwen3EmbeddingClient(qcfg)

    # Domain
    print("[3] Load domain embeddings (aligned) ...")
    E_dom = load_domain_aligned(args.stage2_dir, master_hpo_ids)
    print("    E_dom:", E_dom.shape)
    np.save(os.path.join(args.out_dir, "E_dom.npy"), E_dom)

    # Def
    print("[4] Load/build def embeddings ...")
    E_def, def_src = build_def_embeddings(
        def_embed_dir,
        master_hpo_ids,
        hpo2name,
        qwen_client=qwen,
        qwen_mode="doc",
        force_reencode=bool(args.force_def_reencode),
    )
    print("    E_def:", E_def.shape, f"({def_src})")
    np.save(os.path.join(args.out_dir, "E_def.npy"), E_def)

    # Med / Weak prototypes
    stats = {}
    P_med = n_med = None
    if args.use_med:
        print("[5] Build med prototypes (STRICT mask) ...")
        _ensure_dir(med_embed_dir, "Medium_embed")
        pE = os.path.join(med_embed_dir, "E_med_lines.npy")
        pmask = os.path.join(med_embed_dir, "mask_med.npy")
        pidx = os.path.join(med_embed_dir, "med_index.json")
        _ensure_file(pE, "E_med_lines.npy")
        _ensure_file(pmask, "mask_med.npy")
        _ensure_file(pidx, "med_index.json")

        E_med = np.load(pE).astype(np.float32, copy=False)
        mask_med = np.load(pmask)  # expected [N_master]
        ranges_med = load_index_ranges(pidx)

        P_med, n_med, s_med = build_pool_prototypes_strict_mask(
            pool_name="med",
            master_hpo_ids=master_hpo_ids,
            E_pool=E_med,
            mask_hpo=mask_med,
            ranges=ranges_med,
            topk=args.med_topk,
        )
        stats["med"] = s_med
        np.save(os.path.join(args.out_dir, "prototypes_med.npy"), P_med)
        np.save(os.path.join(args.out_dir, "counts_med.npy"), n_med)

    P_weak = n_weak = None
    if args.use_weak:
        print("[6] Build weak prototypes (STRICT mask) ...")
        _ensure_dir(weak_embed_dir, "weak")
        pE = os.path.join(weak_embed_dir, "E_weak_docs.npy")
        pmask = os.path.join(weak_embed_dir, "mask_weak.npy")
        pidx = os.path.join(weak_embed_dir, "weak_index.json")
        _ensure_file(pE, "E_weak_docs.npy")
        _ensure_file(pmask, "mask_weak.npy")
        _ensure_file(pidx, "weak_index.json")

        E_weak = np.load(pE).astype(np.float32, copy=False)
        mask_weak = np.load(pmask)  # expected [N_master]
        ranges_weak = load_index_ranges(pidx)

        P_weak, n_weak, s_weak = build_pool_prototypes_strict_mask(
            pool_name="weak",
            master_hpo_ids=master_hpo_ids,
            E_pool=E_weak,
            mask_hpo=mask_weak,
            ranges=ranges_weak,
            topk=args.weak_topk,
        )
        stats["weak"] = s_weak
        np.save(os.path.join(args.out_dir, "prototypes_weak.npy"), P_weak)
        np.save(os.path.join(args.out_dir, "counts_weak.npy"), n_weak)

    if stats:
        write_json(os.path.join(args.out_dir, "prototype_stats.json"), stats)

    # Fuse
    print("[7] Fuse vectors V0 ...")
    V0 = fuse_vectors(
        E_def, E_dom,
        lam=args.lam,
        use_med=args.use_med, P_med=P_med, n_med=n_med, K_med_sat=args.K_med_sat, alpha=args.alpha,
        use_weak=args.use_weak, P_weak=P_weak, n_weak=n_weak, K_weak_sat=args.K_weak_sat, beta=args.beta,
    )
    print("    V0:", V0.shape)
    np.save(os.path.join(args.out_dir, "V0_fused.npy"), V0)

    # Round 1 kNN idx
    print("[8] Round-1 HNSW kNN idx (K=%d) ..." % int(args.K))
    idx1 = hnsw_knn_idx_only(V0, K=args.K, M=args.M, efC=args.efC, efS=args.efS, seed=args.seed)
    np.save(os.path.join(args.out_dir, "knn_round1_idx.npy"), idx1)

    print("[9] Round-1 recompute exact sim for edges ...")
    sim1 = recompute_knn_sim_from_idx(V0, idx1, desc="SIM1 exact dot")
    np.save(os.path.join(args.out_dir, "knn_round1_sim_exact.npy"), sim1)

    print("[10] Round-1 build graph + Leiden ...")
    g1 = knn_to_igraph(idx1, sim1, sym="max", sim_min=args.sim_min)
    labels1 = leiden_cluster(g1, resolution=args.resolution, seed=args.seed)
    ncl1 = int(labels1.max()) + 1
    print(f"    clusters_round1 = {ncl1}")
    np.save(os.path.join(args.out_dir, "labels_round1.npy"), labels1)
    write_json(os.path.join(args.out_dir, "cluster_sizes_round1.json"), cluster_sizes(labels1))
    write_clusters_jsonl(
        os.path.join(args.out_dir, "clusters_round1.jsonl"),
        master_hpo_ids, hpo2name, labels1
    )

    # Smooth
    print("[11] Cluster-aware smoothing (eta=%.4f) ..." % float(args.eta))
    V1 = cluster_smooth(V0, labels1, eta=args.eta)
    print("    V1:", V1.shape)
    np.save(os.path.join(args.out_dir, "V1_smooth.npy"), V1)

    # Round 2 kNN idx
    print("[12] Round-2 HNSW kNN idx (K=%d) ..." % int(args.K))
    idx2 = hnsw_knn_idx_only(V1, K=args.K, M=args.M, efC=args.efC, efS=args.efS, seed=args.seed)
    np.save(os.path.join(args.out_dir, "knn_round2_idx.npy"), idx2)

    print("[13] Round-2 recompute exact sim for edges ...")
    sim2 = recompute_knn_sim_from_idx(V1, idx2, desc="SIM2 exact dot")
    np.save(os.path.join(args.out_dir, "knn_round2_sim_exact.npy"), sim2)

    print("[14] Round-2 build graph + Leiden ...")
    g2 = knn_to_igraph(idx2, sim2, sym="max", sim_min=args.sim_min)
    labels2 = leiden_cluster(g2, resolution=args.resolution, seed=args.seed)
    ncl2 = int(labels2.max()) + 1
    print(f"    clusters_round2 = {ncl2}")
    np.save(os.path.join(args.out_dir, "labels_round2.npy"), labels2)
    write_json(os.path.join(args.out_dir, "cluster_sizes_round2.json"), cluster_sizes(labels2))
    write_clusters_jsonl(
        os.path.join(args.out_dir, "clusters_round2.jsonl"),
        master_hpo_ids, hpo2name, labels2
    )

    # Summary
    summary = {
        "created_at": now_ts(),
        "N_master": N,
        "D": int(V0.shape[1]),
        "K": int(args.K),
        "resolution": float(args.resolution),
        "eta": float(args.eta),
        "clusters_round1": ncl1,
        "clusters_round2": ncl2,
        "def_source": def_src,
        "use_med": bool(args.use_med),
        "use_weak": bool(args.use_weak),
        "outputs": {
            "E_dom": "E_dom.npy",
            "E_def": "E_def.npy",
            "V0_fused": "V0_fused.npy",
            "V1_smooth": "V1_smooth.npy",
            "idx1": "knn_round1_idx.npy",
            "sim1": "knn_round1_sim_exact.npy",
            "labels1": "labels_round1.npy",
            "clusters1_jsonl": "clusters_round1.jsonl",
            "idx2": "knn_round2_idx.npy",
            "sim2": "knn_round2_sim_exact.npy",
            "labels2": "labels_round2.npy",
            "clusters2_jsonl": "clusters_round2.jsonl",
        },
    }
    if stats:
        summary["prototype_stats"] = stats
    write_json(os.path.join(args.out_dir, "summary.json"), summary)

    print("[DONE] Saved full reclustering outputs to:")
    print("  ", os.path.abspath(args.out_dir))
    print("[FILES] summary.json / meta.json / clusters_round*.jsonl / V*.npy / knn_round*.npy / labels_round*.npy")


if __name__ == "__main__":
    main()
