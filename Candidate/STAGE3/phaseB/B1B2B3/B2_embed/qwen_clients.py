#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qwen_clients.py  (TRANSFORMERS-ONLY VERSION)

You asked: "use transformers usage (AutoTokenizer/AutoModel) and rewrite the entire client code".

This file provides:
1) Qwen3EmbeddingClient:
   - Loads local Qwen3-Embedding-8B from:
       /cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B
   - Implements the official-style "instruction + query" formatting
   - Uses LAST-TOKEN pooling with left-padding detection (as in your snippet)
   - Produces L2-normalized embeddings suitable for cosine similarity / ANN

2) QwenRerankerClient:
   - Loads local Qwen3-Reranker-4B from:
       /cluster/home/gw/Backend_project/models/Qwen3-Reranker-4B
   - Uses CAUSAL-LM yes/no margin scoring (safe default)
   - Fixes "batch>1 but no padding token" by setting tokenizer + model pad_token_id

Design goals:
- No sentence-transformers dependency.
- HPC-safe batching, device/dtype selection, no flash_attn requirement by default.
- Optional: enable flash_attention_2 if installed.

Requirements:
- transformers>=4.51.0 (you have 4.53.3)
- torch

Notes:
- Embedding max_length can be large (Qwen3-Embedding-8B supports long context),
  but huge values will increase latency/memory. Use 512-2048 for short phrases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)


# -------------------------
# Small utils
# -------------------------

def _ensure_dir(path: str, what: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_dtype(dtype: Optional[str] = None) -> torch.dtype:
    if dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    d = dtype.lower()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def _chunk_list(xs: List[Any], bs: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def _ensure_pad_token(tokenizer, model=None) -> None:
    """
    Ensure pad token exists for tokenizer AND model.config.pad_token_id.
    Avoids Qwen3 forward error for batch>1.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if model is not None:
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # If tokenizer expanded, resize model embeddings
        try:
            n_tok = len(tokenizer)
            n_emb = model.get_input_embeddings().num_embeddings
            if n_tok > n_emb:
                model.resize_token_embeddings(n_tok)
        except Exception:
            pass


# -------------------------
# Embedding: last-token pooling (official snippet)
# -------------------------

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    last_hidden_states: [B, T, H]
    attention_mask:     [B, T]  (1 for tokens, 0 for pad)
    Detect left padding: if last position in every row is non-pad (sum == batch_size),
    then padding is on the left, so last token is the pooled token.
    Otherwise, pooled token is the last non-pad token per row.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


# -------------------------
# Embedding client
# -------------------------

@dataclass
class EmbeddingConfig:
    model_dir: str = "/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B"
    device: Optional[str] = None
    dtype: Optional[str] = None  # "bf16" / "fp16" / "fp32"
    batch_size: int = 32
    max_length: int = 2048
    normalize: bool = True

    # tokenizer/model settings
    padding_side: str = "left"
    trust_remote_code: bool = True

    # Optional acceleration (requires flash_attn installed)
    attn_implementation: Optional[str] = None  # "flash_attention_2" or None

    # Query formatting
    use_instruct: bool = True
    task_description: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    instruct_prefix: str = "Instruct: "
    query_prefix: str = "Query:"


class Qwen3EmbeddingClient:
    """
    Local embedding client for Qwen3-Embedding-8B using transformers.

    encode_queries(queries) -> [N, H]
    encode_docs(docs)       -> [N, H]
    encode(texts, mode=...) -> [N, H]
    similarity(q_emb, d_emb)-> cosine sim matrix [Nq, Nd]
    """

    def __init__(self, cfg: EmbeddingConfig = EmbeddingConfig()):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.dtype = _pick_dtype(cfg.dtype)

        _ensure_dir(cfg.model_dir, "Embedding model_dir")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_dir,
            trust_remote_code=cfg.trust_remote_code,
            padding_side=cfg.padding_side,
            use_fast=True,
        )

        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=cfg.trust_remote_code,
            device_map=None,
        )
        if cfg.attn_implementation:
            # Will crash if flash_attn not installed; keep default None unless you have it.
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModel.from_pretrained(cfg.model_dir, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        _ensure_pad_token(self.tokenizer, self.model)

    def _format_query(self, q: str) -> str:
        if not self.cfg.use_instruct:
            return q
        # Same as your snippet:
        # Instruct: {task}\nQuery:{query}
        return f"{self.cfg.instruct_prefix}{self.cfg.task_description}\n{self.cfg.query_prefix}{q}"

    def _prepare_texts(self, texts: List[str], mode: str) -> List[str]:
        """
        mode:
          - "query": apply instruction formatting
          - "doc": no instruction
          - "raw": no instruction
        """
        if mode == "query":
            return [self._format_query(t) for t in texts]
        return texts

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        *,
        mode: str = "raw",               # "query" / "doc" / "raw"
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        return_numpy: bool = False,
    ):
        if not texts:
            raise ValueError("encode(): texts is empty")
        bs = batch_size or self.cfg.batch_size
        ml = max_length or self.cfg.max_length
        norm = self.cfg.normalize if normalize is None else normalize

        texts_ = self._prepare_texts(texts, mode=mode)

        vecs: List[torch.Tensor] = []
        for batch in _chunk_list(texts_, bs):
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            outputs = self.model(**batch_dict, return_dict=True)
            pooled = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

            if norm:
                pooled = F.normalize(pooled, p=2, dim=1)

            vecs.append(pooled.detach().float().cpu())

        mat = torch.cat(vecs, dim=0)  # [N, H]
        if return_numpy:
            return mat.numpy()
        return mat

    @torch.no_grad()
    def encode_queries(self, queries: List[str], **kwargs):
        return self.encode(queries, mode="query", **kwargs)

    @torch.no_grad()
    def encode_docs(self, docs: List[str], **kwargs):
        return self.encode(docs, mode="doc", **kwargs)

    @staticmethod
    def similarity(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity matrix using normalized embeddings.
        If embeddings are normalized, dot product equals cosine sim.
        """
        if query_emb.dim() != 2 or doc_emb.dim() != 2:
            raise ValueError("similarity expects 2D tensors")
        return query_emb @ doc_emb.T


# -------------------------
# Reranker client (CausalLM yes/no margin)
# -------------------------

@dataclass
class RerankerConfig:
    model_dir: str = "/cluster/home/gw/Backend_project/models/Qwen3-Reranker-4B"
    device: Optional[str] = None
    dtype: Optional[str] = None  # "bf16" / "fp16" / "fp32"
    batch_size: int = 8
    max_length: int = 512
    trust_remote_code: bool = True

    padding_side: str = "left"
    attn_implementation: Optional[str] = None  # "flash_attention_2" or None

    yes_token: str = "yes"
    no_token: str = "no"


class QwenRerankerClient:
    """
    Local reranker client:
    - Uses AutoModelForCausalLM for stable loading (avoids random SeqCls heads)
    - score = logit(yes) - logit(no) at last position
    """

    def __init__(self, cfg: RerankerConfig = RerankerConfig()):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.dtype = _pick_dtype(cfg.dtype)

        _ensure_dir(cfg.model_dir, "Reranker model_dir")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_dir,
            trust_remote_code=cfg.trust_remote_code,
            padding_side=cfg.padding_side,
            use_fast=True,
        )

        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=cfg.trust_remote_code,
            device_map=None,
        )
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        _ensure_pad_token(self.tokenizer, self.model)

        self._yes_id = self._single_token_id(cfg.yes_token)
        self._no_id = self._single_token_id(cfg.no_token)

    def _single_token_id(self, tok: str) -> int:
        ids = self.tokenizer.encode(tok, add_special_tokens=False)
        if not ids:
            return int(self.tokenizer.eos_token_id)
        return int(ids[0])

    def _build_pairs(self, pairs: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        qs = [q for q, _ in pairs]
        cs = [c for _, c in pairs]
        enc = self.tokenizer(
            qs,
            cs,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        if "attention_mask" not in enc:
            enc["attention_mask"] = (enc["input_ids"] != self.tokenizer.pad_token_id).long()
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def score_pairs(self, pairs: List[Tuple[str, str]], *, batch_size: Optional[int] = None) -> List[float]:
        if not pairs:
            return []
        bs = batch_size or self.cfg.batch_size

        scores: List[float] = []
        for batch in _chunk_list(pairs, bs):
            enc = self._build_pairs(batch)
            out = self.model(**enc, return_dict=True)
            last = out.logits[:, -1, :]  # [B, V]
            sc = last[:, self._yes_id] - last[:, self._no_id]
            scores.extend(sc.detach().float().cpu().tolist())
        return scores

    @torch.no_grad()
    def score(self, query: str, candidates: List[str], *, batch_size: Optional[int] = None) -> List[float]:
        return self.score_pairs([(query, c) for c in candidates], batch_size=batch_size)


# -------------------------
# Quick self-test
# -------------------------

if __name__ == "__main__":
    # Embedding quick test (matches official style)
    emb = Qwen3EmbeddingClient(EmbeddingConfig(
        model_dir="/cluster/home/gw/Backend_project/models/Qwen3-Embedding-8B",
        batch_size=8,
        max_length=512,          # short phrases => 512 is enough
        attn_implementation=None,  # set "flash_attention_2" only if flash_attn installed
        use_instruct=True,
    ))

    task = "Given a web search query, retrieve relevant passages that answer the query"
    emb.cfg.task_description = task

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    q_emb = emb.encode_queries(queries)
    d_emb = emb.encode_docs(documents)
    scores = emb.similarity(q_emb, d_emb)
    print("scores:\n", scores)

    # Reranker quick test
    rr = QwenRerankerClient(RerankerConfig(
        model_dir="/cluster/home/gw/Backend_project/models/Qwen3-Reranker-4B",
        batch_size=2,
        max_length=256,
        attn_implementation=None,
    ))
    q = "limited wrist extension"
    cands = [
        "wrist extensor tendon contracture and joint stiffness",
        "renal dysplasia and hydronephrosis",
    ]
    print("rerank scores:", rr.score(q, cands))
