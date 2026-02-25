#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stageB2_recluster_test_local_qwen.py  (REVISED)

Key change you requested:
- Use HNSW ONLY to get neighbor indices (idx).
- Recompute edge weights by exact dot: w(i,j)=float(V[i] @ V[j]) (cosine if V is L2-normalized).
- Feed recomputed weights into Leiden.

This fixes the "HNSW sim scale mismatch" problem and makes graph weights consistent with exact cosine.

Deps:
  pip install numpy hnswlib igraph leidenalg
Optional:
  pip install faiss-cpu (not required here)

Run:
  python3 stageB2_recluster_test_local_qwen.py \
    --b2_out_dir /cluster/home/gw/.../B2_embed/out \
    --stage2_dir /cluster/home/gw/.../STAGE2/cluster_data/frozen_E1K10_20260119_160424 \
    --target_hpo HP:0000002 \
    --K 50 \
    --topk_test 20 \
    --use_med --use_weak
"""

from __future__ import annotations

import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import hnswlib
import igraph as ig
import leidenalg

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


# =============================================================================
# Basic helpers
# =============================================================================

def _ensure_file(p: str, what: str) -> None:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{what} not found: {p}")

def _ensure_dir(p: str, what: str) -> None:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{what} not found: {p}")

def read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

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

def resolve_b2_out_dir(b2_out_dir: str) -> str:
    """
    Accept either:
      - .../B2_embed/out
      - .../B2_embed  (auto append /out if exists)
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

def load_hpo_ids_fallback_jsonl(p: str) -> List[str]:
    """
    If hpo_ids.json is not valid JSON or not in recognized formats,
    try JSONL parsing where each line may contain {"hpo_id": "..."} or a raw string.
    """
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


# =============================================================================
# Load master ids + names
# =============================================================================

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
# Stage2 domain embeddings aligned to master order
# =============================================================================

def load_stage2_ids(stage2_ids_json: str) -> List[str]:
    """
    Supports:
      - {"index_type":"row_index","ids":[...]}  (your freezer format)
      - ["HP:..."] list
      - {"ids":[...]}
    """
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

    E = np.load(pE).astype(np.float32, copy=False)   # [N2, D]
    ids = load_stage2_ids(pids)                      # [N2]
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
# Def embeddings
# =============================================================================

def build_def_embeddings(
    def_embed_dir: str,
    master_hpo_ids: List[str],
    hpo2name: Dict[str, str],
    *,
    qwen_client: Qwen3EmbeddingClient,
    qwen_mode: str = "doc",
) -> Tuple[np.ndarray, str]:
    """
    Priority:
      1) Use Def_embed/E_def.npy if present and rows match N
      2) Else build via Qwen client over text = "{name} ({hpo_id})"
    Returns: (E_def, source_str)
    """
    pE = os.path.join(def_embed_dir, "E_def.npy")
    if os.path.isfile(pE):
        E = np.load(pE).astype(np.float32, copy=False)
        if E.shape[0] == len(master_hpo_ids):
            return l2_normalize_mat(E), "loaded E_def.npy"

    def _make_text(hpo_id: str) -> str:
        name = hpo2name.get(hpo_id, "")
        if name:
            return f"{name} ({hpo_id})"
        return hpo_id

    texts = [_make_text(h) for h in master_hpo_ids]
    E = qwen_client.encode(texts, mode=qwen_mode, return_numpy=True)
    E = np.asarray(E, dtype=np.float32)
    return l2_normalize_mat(E), "qwen-encoded(name+hpo_id)"


# =============================================================================
# Med / Weak prototypes with STRICT mask usage (never ignore)
# =============================================================================

def load_index_ranges(p_index_json: str) -> Dict[str, Tuple[int, int]]:
    """
    med_index.json / weak_index.json maps hpo_id -> [s, e] or {"s":..,"e":..}
    """
    idx = read_json(p_index_json)
    out: Dict[str, Tuple[int, int]] = {}
    for k, v in idx.items():
        if isinstance(v, list) and len(v) == 2:
            out[k] = (int(v[0]), int(v[1]))
        elif isinstance(v, dict) and "s" in v and "e" in v:
            out[k] = (int(v["s"]), int(v["e"]))
    return out

def topk_mean_pool(rows: np.ndarray, k: int) -> np.ndarray:
    """
    rows: [m, D], assumed L2-normalized rows.
    centroid -> score -> topk -> mean -> normalize
    """
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
    mask_hpo: np.ndarray,   # shape [N_master], True means "this HPO has pool evidence"
    ranges: Dict[str, Tuple[int, int]],
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STRICT interpretation:
    - mask is HPO-level (len == N_master), not row-level.
    - We NEVER ignore it.
    - If mask_hpo[i] is False -> prototype stays zero and count stays 0 even if ranges says otherwise.
    - If mask_hpo[i] is True -> we use ranges[hpo] to slice E_pool rows and pool.
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

    # Debug
    print(f"[DEBUG {pool_name}] E_pool.shape    =", E_pool.shape)
    print(f"[DEBUG {pool_name}] mask.shape      =", mask_hpo.shape)
    print(f"[DEBUG {pool_name}] mask.ndim       =", mask_hpo.ndim)
    print(f"[DEBUG {pool_name}] ranges size     =", len(ranges))
    some = next(iter(ranges.items()))
    print(f"[DEBUG {pool_name}] example range   =", some)

    # Build
    for i, h in enumerate(master_hpo_ids):
        if not bool(mask_hpo[i]):
            continue
        if h not in ranges:
            continue
        s, e = ranges[h]
        if e <= s:
            continue
        if s < 0 or e > E_pool.shape[0]:
            # Hard error: your index/range points outside pool rows => data corruption
            raise ValueError(
                f"{pool_name} range out of bounds for {h}: (s,e)=({s},{e}) but E_pool rows={E_pool.shape[0]}"
            )
        rows = E_pool[s:e]
        if rows.shape[0] == 0:
            continue
        n[i] = rows.shape[0]
        P[i] = topk_mean_pool(rows, topk)

    return P, n


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

    N, D = E_def.shape
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
# kNN via HNSW (idx only) + exact weight recompute
# =============================================================================

def hnsw_knn_idx_only(
    V: np.ndarray,
    *,
    K: int = 50,
    M: int = 32,
    efC: int = 200,
    efS: int = 200,
) -> np.ndarray:
    """
    Return idx [N,K] from HNSW. No similarity trusted/returned.
    Uses inner product space; V should be L2-normalized.
    """
    V = V.astype(np.float32, copy=False)
    N, D = V.shape
    index = hnswlib.Index(space="ip", dim=D)
    index.init_index(max_elements=N, ef_construction=efC, M=M)
    index.add_items(V, np.arange(N))
    index.set_ef(efS)

    idx, _ = index.knn_query(V, k=K + 1)  # includes self
    idx = idx[:, 1:].astype(np.int32)
    return idx

def recompute_knn_sim_from_idx(V: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Given V [N,D] L2-normalized and idx [N,K],
    return sim [N,K] where sim[i,t] = float(V[i] @ V[idx[i,t]]).
    """
    V = V.astype(np.float32, copy=False)
    N, K = idx.shape
    sim = np.empty((N, K), dtype=np.float32)
    for i in range(N):
        vi = V[i]
        js = idx[i]
        sim[i] = V[js] @ vi
    return sim


# =============================================================================
# Graph build + Leiden (weights are exact dot)
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
    for i in range(N):
        for t in range(K):
            j = int(idx[i, t])
            w = float(sim[i, t])
            if i == j:
                continue
            if w < sim_min:
                continue
            wdir[(i, j)] = w

    edges = []
    weights = []
    seen = set()
    for (i, j), wij in wdir.items():
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
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
        weights.append(w)

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    return g

def leiden_cluster(g: ig.Graph, *, resolution: float = 1.0, seed: int = 42) -> np.ndarray:
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

def cluster_smooth(V: np.ndarray, labels: np.ndarray, *, eta: float = 0.08) -> np.ndarray:
    V = V.astype(np.float32, copy=False)
    labels = labels.astype(np.int32, copy=False)
    N, D = V.shape
    C = int(labels.max()) + 1

    mu = np.zeros((C, D), dtype=np.float32)
    cnt = np.zeros((C,), dtype=np.int32)
    for i in range(N):
        c = int(labels[i])
        mu[c] += V[i]
        cnt[c] += 1
    for c in range(C):
        if cnt[c] > 0:
            mu[c] = l2_normalize_vec(mu[c] / float(cnt[c]))
        else:
            mu[c] = 0.0

    V2 = (1.0 - eta) * V + eta * mu[labels]
    return l2_normalize_mat(V2)


# =============================================================================
# Neighbor test (ANN idx from HNSW; ANN sim from exact recompute)
# =============================================================================

def print_neighbor_test(
    *,
    target_hpo: str,
    master_hpo_ids: List[str],
    hpo2name: Dict[str, str],
    V: np.ndarray,
    idx_ann: np.ndarray,
    sim_ann: np.ndarray,
    topk: int = 20,
) -> None:
    pos = {h: i for i, h in enumerate(master_hpo_ids)}
    if target_hpo not in pos:
        raise ValueError(f"target_hpo not found in master list: {target_hpo}")

    i = pos[target_hpo]
    name = hpo2name.get(target_hpo, "")
    print("=" * 80)
    print(f"[TARGET] {target_hpo} | {name}")
    print("-" * 80)

    print(f"[ANN neighbors] top{topk}  (idx from HNSW, weight=exact dot)")
    ann_js = idx_ann[i, :topk]
    ann_ws = sim_ann[i, :topk]
    for r, (j, w) in enumerate(zip(ann_js, ann_ws), 1):
        hid = master_hpo_ids[int(j)]
        nm = hpo2name.get(hid, "")
        print(f"{r:02d}  sim={float(w):.4f}  {hid}  {nm}")

    print("-" * 80)
    print(f"[EXACT neighbors] top{topk}")
    q = V[i]
    scores = V @ q
    scores[i] = -1.0
    exact_idx = topk_indices(scores, topk)
    for r, j in enumerate(exact_idx, 1):
        hid = master_hpo_ids[int(j)]
        nm = hpo2name.get(hid, "")
        print(f"{r:02d}  sim={float(scores[j]):.4f}  {hid}  {nm}")

    ann_set = set(map(int, ann_js.tolist()))
    ex_set = set(map(int, exact_idx.tolist()))
    inter = ann_set & ex_set
    hit = len(inter) / float(topk)
    print("-" * 80)
    print(f"[OVERLAP] ANN∩EXACT = {len(inter)}/{topk}  hit_rate={hit:.2%}")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2_out_dir", required=True, help=".../B2_embed/out OR .../B2_embed (auto out/)")
    ap.add_argument("--stage2_dir", required=True, help=".../STAGE2/cluster_data/frozen_E1K10_...")

    ap.add_argument("--target_hpo", default="HP:0000002", help="HPO id for neighbor test")
    ap.add_argument("--topk_test", type=int, default=20, help="neighbors to print + validate")

    # Qwen embedding client config
    ap.add_argument("--qwen_model_dir", default="/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B")
    ap.add_argument("--qwen_bs", type=int, default=32)
    ap.add_argument("--qwen_maxlen", type=int, default=512)
    ap.add_argument("--qwen_dtype", default=None, help="bf16/fp16/fp32 or None(auto)")
    ap.add_argument("--qwen_attn_impl", default=None, help='set "flash_attention_2" if installed')

    # Fusion params
    ap.add_argument("--lam", type=float, default=0.6)

    # Optional med/weak integration
    ap.add_argument("--use_med", action="store_true")
    ap.add_argument("--use_weak", action="store_true")
    ap.add_argument("--med_topk", type=int, default=5)
    ap.add_argument("--weak_topk", type=int, default=5)
    ap.add_argument("--K_med_sat", type=int, default=10)
    ap.add_argument("--K_weak_sat", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.05)

    # HNSW
    ap.add_argument("--K", type=int, default=50, help="kNN neighbors for graph")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--efS", type=int, default=200)
    ap.add_argument("--sim_min", type=float, default=-1.0, help="prune edges < sim_min when building graph")

    # Leiden
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Reclustering smoothing
    ap.add_argument("--eta", type=float, default=0.08)

    args = ap.parse_args()

    # Resolve dirs
    _ensure_dir(args.b2_out_dir, "b2_out_dir (as provided)")
    _ensure_dir(args.stage2_dir, "stage2_dir")
    args.b2_out_dir = resolve_b2_out_dir(args.b2_out_dir)

    def_embed_dir = os.path.join(args.b2_out_dir, "Def_embed")
    med_embed_dir = os.path.join(args.b2_out_dir, "Medium_embed")
    weak_embed_dir = os.path.join(args.b2_out_dir, "weak")

    _ensure_dir(def_embed_dir, "Def_embed")
    def_source = os.path.join(def_embed_dir, "def_source.jsonl")
    _ensure_file(def_source, "def_source.jsonl")

    print("[PATH] resolved b2_out_dir =", args.b2_out_dir)
    print("[PATH] def_embed_dir      =", def_embed_dir)
    print("[PATH] hpo_ids.json       =", os.path.join(def_embed_dir, "hpo_ids.json"))

    # master ids + names
    master_hpo_ids = load_master_hpo_ids(def_embed_dir)
    hpo2name = load_hpo_names(def_source)

    # Qwen client
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

    # Domain aligned
    print("[1] Loading Domain embeddings (aligned)...")
    E_dom = load_domain_aligned(args.stage2_dir, master_hpo_ids)
    print("    E_dom:", E_dom.shape)

    # Def embeddings
    print("[2] Building Def embeddings (E_def.npy if usable; else Qwen encode)...")
    E_def, def_src = build_def_embeddings(def_embed_dir, master_hpo_ids, hpo2name, qwen_client=qwen, qwen_mode="doc")
    print("    E_def:", E_def.shape, f"({def_src})")

    # Optional: Med / Weak prototypes (STRICT mask usage)
    P_med = n_med = None
    if args.use_med:
        _ensure_dir(med_embed_dir, "Medium_embed")
        pE = os.path.join(med_embed_dir, "E_med_lines.npy")
        pmask = os.path.join(med_embed_dir, "mask_med.npy")
        pidx = os.path.join(med_embed_dir, "med_index.json")
        _ensure_file(pE, "E_med_lines.npy")
        _ensure_file(pmask, "mask_med.npy")
        _ensure_file(pidx, "med_index.json")
        print("[3] Building Med prototypes (STRICT mask)...")
        E_med = np.load(pE).astype(np.float32, copy=False)
        mask_med = np.load(pmask)  # expected shape [N_master]
        ranges_med = load_index_ranges(pidx)
        P_med, n_med = build_pool_prototypes_strict_mask(
            pool_name="med",
            master_hpo_ids=master_hpo_ids,
            E_pool=E_med,
            mask_hpo=mask_med,
            ranges=ranges_med,
            topk=args.med_topk,
        )

    P_weak = n_weak = None
    if args.use_weak:
        _ensure_dir(weak_embed_dir, "weak")
        pE = os.path.join(weak_embed_dir, "E_weak_docs.npy")
        pmask = os.path.join(weak_embed_dir, "mask_weak.npy")
        pidx = os.path.join(weak_embed_dir, "weak_index.json")
        _ensure_file(pE, "E_weak_docs.npy")
        _ensure_file(pmask, "mask_weak.npy")
        _ensure_file(pidx, "weak_index.json")
        print("[4] Building Weak prototypes (STRICT mask)...")
        E_weak = np.load(pE).astype(np.float32, copy=False)
        mask_weak = np.load(pmask)  # expected shape [N_master]
        ranges_weak = load_index_ranges(pidx)
        P_weak, n_weak = build_pool_prototypes_strict_mask(
            pool_name="weak",
            master_hpo_ids=master_hpo_ids,
            E_pool=E_weak,
            mask_hpo=mask_weak,
            ranges=ranges_weak,
            topk=args.weak_topk,
        )

    # Fuse
    print("[5] Fusing vectors v(h)...")
    V = fuse_vectors(
        E_def, E_dom,
        lam=args.lam,
        use_med=args.use_med, P_med=P_med, n_med=n_med, K_med_sat=args.K_med_sat, alpha=args.alpha,
        use_weak=args.use_weak, P_weak=P_weak, n_weak=n_weak, K_weak_sat=args.K_weak_sat, beta=args.beta,
    )

    # Round 1: HNSW idx only + recompute weights by exact dot
    print("[6] Round-1 kNN backend = hnsw (idx only) + exact weight recompute ...")
    idx1 = hnsw_knn_idx_only(V, K=args.K, M=args.M, efC=args.efC, efS=args.efS)
    sim1 = recompute_knn_sim_from_idx(V, idx1)

    print("[7] Round-1 Leiden (weights=exact dot)...")
    g1 = knn_to_igraph(idx1, sim1, sym="max", sim_min=args.sim_min)
    labels1 = leiden_cluster(g1, resolution=args.resolution, seed=args.seed)
    print(f"    clusters_round1 = {int(labels1.max()) + 1}")

    # Reclustering smoothing
    print("[8] Reclustering (cluster-aware smoothing) ...")
    V2 = cluster_smooth(V, labels1, eta=args.eta)

    # Round 2: HNSW idx only + recompute weights by exact dot
    print("[9] Round-2 kNN backend = hnsw (idx only) + exact weight recompute ...")
    idx2 = hnsw_knn_idx_only(V2, K=args.K, M=args.M, efC=args.efC, efS=args.efS)
    sim2 = recompute_knn_sim_from_idx(V2, idx2)

    print("[10] Round-2 Leiden (weights=exact dot)...")
    g2 = knn_to_igraph(idx2, sim2, sym="max", sim_min=args.sim_min)
    labels2 = leiden_cluster(g2, resolution=args.resolution, seed=args.seed)
    print(f"    clusters_round2 = {int(labels2.max()) + 1}")

    # Neighbor sanity test
    print_neighbor_test(
        target_hpo=args.target_hpo,
        master_hpo_ids=master_hpo_ids,
        hpo2name=hpo2name,
        V=V2,
        idx_ann=idx2,
        sim_ann=sim2,
        topk=args.topk_test,
    )


if __name__ == "__main__":
    main()
