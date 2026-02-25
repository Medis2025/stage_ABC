#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np

from qwen_clients import Qwen3EmbeddingClient, EmbeddingConfig


HPO_JSON = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json"
MODEL_DIR = "/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B"


def join_list(xs) -> str:
    if xs is None:
        return ""
    if isinstance(xs, str):
        xs = [xs]
    if not isinstance(xs, list):
        xs = [str(xs)]
    parts = []
    for x in xs:
        t = str(x).strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def pick_def_text(
    obj: dict,
    *,
    min_len: int = 60,                 # [CHANGED] 更合理：curated < 60 认为太短
    use_comment: bool = True,
    comment_gate_len: int = 120,
) -> tuple[str, str]:
    """
    Rule (as agreed + refined):
    - Def is list -> join into a paragraph
    - llm_def (if exists) > llm_add_def
    - Comment augmentation ONLY when curated_def is short (optional)
    Always return non-empty text (fallback to Name).
    """
    name = join_list(obj.get("Name") or obj.get("name") or [])
    curated = join_list(obj.get("Def") or obj.get("def") or [])
    comment = join_list(obj.get("Comment") or obj.get("comment") or [])
    llm_def = str(obj.get("llm_def") or "").strip()
    llm_add = str(obj.get("llm_add_def") or "").strip()

    base = curated
    source = "curated_def"

    # [CHANGED] 先记录是否是“短 curated”
    short_curated = (source == "curated_def" and len(base) < min_len)

    if len(base) < min_len:
        if llm_def:
            base, source = llm_def, "llm_def"
        elif llm_add:
            base, source = llm_add, "llm_add_def"
        elif name:
            base, source = name, "name_fallback"
        else:
            base, source = "[EMPTY]", "empty_fallback"

    # [CHANGED] comment 只在“curated_def 且短”时补齐
    if use_comment and comment and short_curated and len(curated) < comment_gate_len:
        base = curated.rstrip(".") + ". " + comment
        source = "curated_def+comment"

    return base, source


def l2_normalize_f32(v: np.ndarray) -> np.ndarray:
    """
    [CHANGED] force float32 + exact L2 normalize on CPU.
    """
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0:
        # defensive fallback: return zeros instead of NaN
        return np.zeros_like(v, dtype=np.float32)
    return (v / (n + 1e-12)).astype(np.float32)


def main():
    if not os.path.exists(HPO_JSON):
        raise FileNotFoundError(f"Missing file: {HPO_JSON}")

    with open(HPO_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    hpo_ids = [k for k in data.keys() if isinstance(k, str) and k.startswith("HP:")]
    if not hpo_ids:
        raise RuntimeError("No HP:* keys found in hpo_enriched_with_llm.json")

    client = Qwen3EmbeddingClient(EmbeddingConfig(
        model_dir=MODEL_DIR,
        batch_size=8,
        max_length=512,
        use_instruct=False,   # def is doc-like text
        normalize=True,       # keep; we'll re-normalize in float32 again
        attn_implementation=None,
    ))

    test_ids = []
    if "HP:0031213" in data:
        test_ids.append("HP:0031213")

    random.seed(42)
    remain = [x for x in hpo_ids if x not in test_ids]
    test_ids += random.sample(remain, k=min(5, len(remain)))

    print(f"[INFO] Loaded {len(hpo_ids)} HPO entries")
    print(f"[INFO] Testing {len(test_ids)} entries: {test_ids}")

    for hid in test_ids:
        obj = data.get(hid, {}) or {}
        name = join_list(obj.get("Name") or obj.get("name") or [])
        text, source = pick_def_text(obj, min_len=60, use_comment=True, comment_gate_len=120)

        emb = client.encode([text], mode="doc", return_numpy=True)  # [1, D]
        v = emb[0]

        # [CHANGED] float32 exact normalize
        v = l2_normalize_f32(v)

        print("\n" + "=" * 90)
        print("HPO:", hid)
        print("Name:", name[:120] + ("..." if len(name) > 120 else ""))
        print("Def source:", source)
        print("Def len:", len(text))
        print("Def preview:", text[:240].replace("\n", " ") + ("..." if len(text) > 240 else ""))

        print("Emb shape:", (1, v.shape[0]))
        print("L2 norm:", float(np.linalg.norm(v)))
        print("Has NaN:", bool(np.isnan(v).any()))
        print("Has Inf:", bool(np.isinf(v).any()))
        print("Self cosine:", float(v @ v))

    print("\n[DONE] def embedding test finished.")


if __name__ == "__main__":
    main()
