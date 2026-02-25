#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
b2_llm_client.py

B2 LLM client (API-only) for "evidence -> unit" compression.
- Strict JSON output
- OpenAI-compatible endpoint (e.g., DeepSeek)
- No local LLM
"""

from __future__ import annotations

import os
import json
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


@dataclass
class B2LLMConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: int = 60
    temperature: float = 0.0
    max_tokens: int = 512
    # If your provider supports response_format json_schema, you can extend later.
    # For now we enforce JSON by prompt + post-parse.


class B2LLMClient:
    """
    Minimal OpenAI-compatible chat client for B2 unit compression.

    Returns Python dict parsed from JSON (strict mode).
    """

    def __init__(self, cfg: B2LLMConfig):
        if not cfg.api_key:
            raise ValueError("api_key is required")
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.cfg.timeout)
        r.raise_for_status()
        data = r.json()

        txt = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Strict JSON parse (allow the model to wrap with ```json ...``` sometimes)
        txt = self._strip_code_fence(txt)
        try:
            return json.loads(txt)
        except Exception as e:
            raise ValueError(f"LLM did not return valid JSON.\nRaw:\n{txt}\nError: {e}")

    @staticmethod
    def _strip_code_fence(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # remove first fence line
            lines = s.splitlines()
            # drop first line
            lines = lines[1:]
            # drop last fence if exists
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return s
