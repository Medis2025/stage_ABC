#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
freeze_embedding_clusters.py

Freeze embedding (E1_name_domain) with K=10 as a system constant and persist:

Layer 0: HPO master / input snapshot
  - copy queries_refilled.jsonl into cluster_data/layer0_hpo_master/

Layer 1: Embedding index
  - E1 vectors saved as .npy (float32, L2-normalized)
  - ids.json stores index->hpo_id mapping + config for reproducibility

Layer 2: Frozen kNN clusters (E1, K=10)
  - neighbors_idx.npy  [N,K]  (int64 indices)
  - neighbors_sim.npy  [N,K]  (float32 cosine)
  - neighbors.jsonl    one line per seed_hpo with neighbors + cosine
  - meta.json          frozen constant declaration

Default dirs (your requested base):
  /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/cluster_data

Run example:
  python freeze_embedding_clusters.py \
    --stage1_jsonl /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/queries_refilled.jsonl \
    --cluster_root /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/cluster_data \
    --model_dir /cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B \
    --batch_size 32 \
    --max_length 512 \
    --k 10 \
    --use_faiss

Notes:
- Embedding view is fixed: E1 = mean( embed([name] + scale_4_domain phrases) )
- scale_4_domain phrases are taken from queries_refilled.jsonl as "scale_4_domain"
- Vectors are L2-normalized so FAISS IP == cosine similarity.
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Iterable, Tuple

import numpy as np
from tqdm import tqdm

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
from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig  # noqa: E402


# =============================================================================
# utils
# =============================================================================

_WS = re.compile(r"\s+")

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

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

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
        xs = [str(xs)]
    out: List[str] = []
    for x in xs:
        t = clean_phrase(str(x))
        if t:
            out.append(t)
    return out

def _chunk_list(xs: List[Any], bs: int) -> Iterable[List[Any]]:
    bs = max(1, int(bs))
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


# =============================================================================
# Layer 0 / Data pack
# =============================================================================

@dataclass
class DataPack:
    ids: List[str]
    names: List[str]
    s4: List[List[str]]  # scale_4_domain

def build_datapack(stage1_items: List[Dict[str, Any]], limit: int = 0) -> DataPack:
    ids, names, s4s = [], [], []
    it = stage1_items[:limit] if limit and limit > 0 else stage1_items
    for obj in it:
        hid = obj.get("hpo_id") or ""
        nm = obj.get("name") or ""
        if not hid or not nm:
            continue
        ids.append(str(hid).strip())
        names.append(clean_phrase(str(nm)))
        s4s.append(nonempty_list(obj.get("scale_4_domain")))
    return DataPack(ids=ids, names=names, s4=s4s)


# =============================================================================
# Layer 1 / E1 embedding (name + domain)
# =============================================================================

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
    )  # [M,D], each already normalized
    v = emb.mean(axis=0, keepdims=False)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)

def build_E1_name_domain(
    client: Qwen3EmbeddingClient,
    dp: DataPack,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """
    E1 = mean( embed([name] + scale_4_domain) ) for each HPO.
    Output: [N, D] float32 L2-normalized.
    """
    N = len(dp.ids)
    if N == 0:
        raise RuntimeError("No valid HPO items found from stage1 jsonl.")
    vecs: List[np.ndarray] = []
    p = tqdm(total=N, desc="EMB E1_name_domain", unit="term")
    for i in range(N):
        phrases = [dp.names[i]] + dp.s4[i]
        # de-dup keep order
        uniq: List[str] = []
        seen = set()
        for x in phrases:
            x = clean_phrase(x)
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        if not uniq:
            uniq = [dp.names[i]]
        v = mean_of_phrase_embs(client, uniq, max_length=max_length, batch_size=batch_size)
        vecs.append(v)
        p.update(1)
    p.close()
    E1 = np.stack(vecs, axis=0).astype(np.float32)
    return E1


# =============================================================================
# Layer 2 / kNN clustering
# =============================================================================

def build_knn_index(vectors: np.ndarray, use_faiss: bool = True) -> Any:
    """
    vectors: [N, D] float32, already normalized.
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

def compute_neighbors_topk(
    ids: List[str],
    vectors: np.ndarray,
    k: int,
    use_faiss: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      neigh_idx: [N,k] neighbor indices (self removed)
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
        out_i: List[int] = []
        out_s: List[float] = []
        for j, sc in zip(row_i, row_s):
            if j == i:
                continue
            out_i.append(int(j))
            out_s.append(float(sc))
            if len(out_i) >= k:
                break
        if len(out_i) < k:
            # pad (rare)
            out_i += [i] * (k - len(out_i))
            out_s += [1.0] * (k - len(out_s))
        neigh_idx[i, :] = np.array(out_i, dtype=np.int64)
        neigh_sim[i, :] = np.array(out_s, dtype=np.float32)

    return neigh_idx, neigh_sim


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser("freeze_embedding_clusters")
    ap.add_argument(
        "--stage1_jsonl",
        type=str,
        default="/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/queries_refilled.jsonl",
        help="Input Stage1 refilled jsonl (Layer 0 source)",
    )
    ap.add_argument(
        "--cluster_root",
        type=str,
        default="/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/llm_corpus/Candidate/Candidate/cluster_data",
        help="Base dir to save Layer0/1/2",
    )
    ap.add_argument("--tag", type=str, default="", help="Optional subdir tag under cluster_root; default=frozen_E1K10_YYYYmmdd_HHMMSS")

    ap.add_argument("--model_dir", type=str, default="/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B")
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / cuda:0 ... (empty=auto)")
    ap.add_argument("--dtype", type=str, default="", help="bf16/fp16/fp32 (empty=auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N items")

    ap.add_argument("--k", type=int, default=10, help="Frozen K for neighbors (default=10)")
    ap.add_argument("--use_faiss", action="store_true", help="Use faiss if available (recommended)")

    args = ap.parse_args()

    # ---- freeze constant (guard)
    if int(args.k) != 10:
        raise ValueError("This freezer is for the system constant: K must be 10. (Pass --k 10)")

    tag = args.tag.strip() or f"frozen_E1K10_{now_tag()}"
    out_root = os.path.join(args.cluster_root, tag)

    # layer dirs
    layer0_dir = os.path.join(out_root, "layer0_hpo_master")
    layer1_dir = os.path.join(out_root, "layer1_embedding_E1_name_domain")
    layer2_dir = os.path.join(out_root, "layer2_clusters_E1_K10")

    ensure_dir(layer0_dir)
    ensure_dir(layer1_dir)
    ensure_dir(layer2_dir)

    # ----------------------------
    # Layer 0: snapshot Stage1 jsonl
    # ----------------------------
    src_jsonl = args.stage1_jsonl
    if not os.path.exists(src_jsonl):
        raise FileNotFoundError(f"stage1_jsonl not found: {src_jsonl}")

    dst_jsonl = os.path.join(layer0_dir, "queries_refilled.jsonl")
    # copy (overwrite to keep run self-contained)
    shutil.copyfile(src_jsonl, dst_jsonl)

    # Load (from snapshot)
    tqdm.write("[L0] loading snapshot jsonl ...")
    stage1_items = read_jsonl(dst_jsonl)
    dp = build_datapack(stage1_items, limit=args.limit)
    N = len(dp.ids)
    if N == 0:
        raise RuntimeError("No usable items from queries_refilled.jsonl (need hpo_id + name).")

    # Save minimal master index (optional but useful)
    master_index_path = os.path.join(layer0_dir, "master_index.json")
    write_json(master_index_path, {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_stage1_jsonl": src_jsonl,
        "snapshot_stage1_jsonl": dst_jsonl,
        "N_used": N,
        "limit": int(args.limit),
        "fields_kept_in_snapshot": "full line objects in jsonl",
        "note": "Layer0 is the single source of semantic text fields (name/scale_2/scale_3/scale_4, etc).",
    })

    # ----------------------------
    # Layer 1: compute E1 embedding and save .npy + ids.json
    # ----------------------------
    tqdm.write("[L1] loading embedding model + building E1_name_domain ...")
    client = Qwen3EmbeddingClient(EmbeddingConfig(
        model_dir=args.model_dir,
        device=(args.device or None),
        dtype=(args.dtype or None),
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=True,
        attn_implementation=None,
        use_instruct=False,
    ))

    E1 = build_E1_name_domain(
        client=client,
        dp=dp,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    # Save vectors
    vec_path = os.path.join(layer1_dir, "E1_name_domain.npy")
    np.save(vec_path, E1)

    # Save id mapping + config
    ids_path = os.path.join(layer1_dir, "ids.json")
    cfg_path = os.path.join(layer1_dir, "config.json")
    write_json(ids_path, {
        "index_type": "row_index",
        "ids": dp.ids,
    })
    write_json(cfg_path, {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_type": "E1_name_domain",
        "definition": "E1 = mean( embed([name] + scale_4_domain phrases) ), then L2 normalize",
        "model_dir": args.model_dir,
        "device": args.device or "auto",
        "dtype": args.dtype or "auto",
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "N": int(E1.shape[0]),
        "D": int(E1.shape[1]),
        "normalize": True,
        "stage1_snapshot": dst_jsonl,
    })

    # ----------------------------
    # Layer 2: build KNN (K=10) and save clusters
    # ----------------------------
    tqdm.write("[L2] building frozen KNN clusters (E1, K=10) ...")
    use_faiss = bool(args.use_faiss) and _HAVE_FAISS
    backend = "faiss" if use_faiss else ("sklearn" if _HAVE_SKLEARN else "none")
    if bool(args.use_faiss) and not _HAVE_FAISS:
        tqdm.write("[WARN] --use_faiss requested but faiss not available; fallback to sklearn brute-force.")
    if backend == "none":
        raise RuntimeError("No retrieval backend available: install faiss (preferred) or sklearn.")

    neigh_idx, neigh_sim = compute_neighbors_topk(dp.ids, E1, k=10, use_faiss=use_faiss)

    idx_path = os.path.join(layer2_dir, "neighbors_idx.npy")
    sim_path = os.path.join(layer2_dir, "neighbors_sim.npy")
    np.save(idx_path, neigh_idx.astype(np.int64))
    np.save(sim_path, neigh_sim.astype(np.float32))

    # Write jsonl clusters for easy debugging / grep / streaming
    rows: List[Dict[str, Any]] = []
    for i, hid in enumerate(dp.ids):
        neigh_list = []
        for r in range(10):
            j = int(neigh_idx[i, r])
            neigh_list.append({
                "hpo_id": dp.ids[j],
                "cosine": float(neigh_sim[i, r]),
                "rank": r + 1,
            })
        rows.append({
            "seed_hpo": hid,
            "embedding_type": "E1_name_domain",
            "K": 10,
            "neighbors": neigh_list,
        })
    jsonl_path = os.path.join(layer2_dir, "neighbors.jsonl")
    write_jsonl(jsonl_path, rows)

    meta_path = os.path.join(layer2_dir, "meta.json")
    write_json(meta_path, {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_constant": True,
        "embedding_type": "E1_name_domain",
        "K": 10,
        "retrieval_backend": backend,
        "stage1_snapshot": dst_jsonl,
        "layer1_vectors": vec_path,
        "layer1_ids": ids_path,
        "outputs": {
            "neighbors_idx": idx_path,
            "neighbors_sim": sim_path,
            "neighbors_jsonl": jsonl_path,
        },
        "notes": [
            "This Layer2 is content-agnostic: it stores only topology (neighbors) + cosine.",
            "All semantic fields should be resolved via Layer0 (queries_refilled.jsonl).",
        ],
    })

    tqdm.write("[DONE] Frozen constants saved under:")
    tqdm.write(f"  {out_root}")
    tqdm.write("[L0] " + layer0_dir)
    tqdm.write("[L1] " + layer1_dir)
    tqdm.write("[L2] " + layer2_dir)


if __name__ == "__main__":
    main()
