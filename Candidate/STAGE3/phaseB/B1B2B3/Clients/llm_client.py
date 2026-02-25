#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_client.py  (DeepSeek / OpenAI-compatible)

A robust, minimal LLM client for your Stage B1/B2/B3 pipeline.

Features
- ✅ OpenAI-compatible /v1/chat/completions
- ✅ Non-stream + stream (SSE)
- ✅ Strict JSON-mode helper (best-effort, provider-dependent)
- ✅ Retry w/ exponential backoff for transient HTTP errors
- ✅ Timeouts, session reuse, safe parsing

Dependencies
- requests

Usage
-----
from llm_client import LLMClient

client = LLMClient(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    timeout=60.0,
)

# 1) Normal completion (string)
text = client.complete_text(
    system="You are a helpful assistant.",
    user="Write 3 short candidate HPO mention phrases for 'seizure'.",
    temperature=0.2,
)
print(text)

# 2) Streaming completion (yields tokens)
for tok in client.stream_text(
    system="You are a helpful assistant.",
    user="Stream 1 paragraph summary about EEG hyperscanning.",
    temperature=0.3,
):
    print(tok, end="", flush=True)

# 3) JSON completion (returns Python object)
obj = client.complete_json(
    system="Return strict JSON only.",
    user="Return {\"b1\":[],\"b2\":[],\"b3\":[]} with 2 items each.",
    temperature=0.0,
)
print(obj)
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import requests


JsonDict = Dict[str, Any]


@dataclass
class LLMError(Exception):
    message: str
    status_code: Optional[int] = None
    response_text: Optional[str] = None

    def __str__(self) -> str:
        extra = []
        if self.status_code is not None:
            extra.append(f"status={self.status_code}")
        if self.response_text:
            rt = self.response_text
            if len(rt) > 500:
                rt = rt[:500] + "..."
            extra.append(f"response={rt}")
        suffix = (" | " + " ".join(extra)) if extra else ""
        return f"{self.message}{suffix}"


class LLMClient:
    """
    DeepSeek / OpenAI-compatible chat.completions client.

    Default endpoint:
      {base_url}/v1/chat/completions

    Notes:
    - DeepSeek is largely OpenAI-compatible for chat completions.
    - JSON mode support depends on provider; we implement a best-effort wrapper:
      - enforce "Return JSON only" instructions
      - optional response_format={"type":"json_object"} if accepted by the server
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("api_key must be provided")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ---------------------------
    # Public convenience methods
    # ---------------------------

    def complete_text(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra_messages: Optional[List[JsonDict]] = None,
        **kwargs: Any,
    ) -> str:
        """Return the assistant message content as a string."""
        payload = self._build_payload(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra_messages=extra_messages,
            **kwargs,
        )
        data = self._post_with_retry("/v1/chat/completions", payload)
        return self._extract_text(data)

    def stream_text(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra_messages: Optional[List[JsonDict]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Stream tokens (best-effort) via SSE.

        Yields incremental token strings. Caller prints/aggregates them.
        """
        payload = self._build_payload(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_messages=extra_messages,
            **kwargs,
        )
        yield from self._post_stream_sse("/v1/chat/completions", payload)

    def complete_json(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_messages: Optional[List[JsonDict]] = None,
        response_format: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Request JSON output and parse it into Python.

        - response_format=True will attempt to pass response_format={"type":"json_object"}
          (provider-dependent). If the server rejects it, you can set response_format=False.

        Returns: parsed Python object (dict/list/etc.)
        """
        sys2 = system.strip()
        guard = (
            "You must return STRICT JSON only. "
            "No markdown, no code fences, no extra keys unless requested."
        )
        system_final = (sys2 + "\n\n" + guard).strip() if sys2 else guard

        payload = self._build_payload(
            system=system_final,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra_messages=extra_messages,
            **kwargs,
        )

        if response_format:
            # provider-dependent (OpenAI supports this; some OpenAI-compatible providers do too)
            payload["response_format"] = {"type": "json_object"}

        data = self._post_with_retry("/v1/chat/completions", payload)
        text = self._extract_text(data).strip()

        # Try strict JSON parse; fallback to extracting first JSON object.
        try:
            return json.loads(text)
        except Exception:
            extracted = self._extract_first_json(text)
            if extracted is None:
                raise LLMError("Model did not return valid JSON", response_text=text)
            try:
                return json.loads(extracted)
            except Exception:
                raise LLMError("Failed to parse extracted JSON", response_text=extracted)

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _build_payload(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        extra_messages: Optional[List[JsonDict]],
        **kwargs: Any,
    ) -> JsonDict:
        msgs: List[JsonDict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if extra_messages:
            # must be OpenAI message dicts
            msgs.extend(extra_messages)
        msgs.append({"role": "user", "content": user})

        payload: JsonDict = {
            "model": self.model,
            "messages": msgs,
            "temperature": float(temperature),
            "stream": bool(stream),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        # pass-through knobs (top_p, presence_penalty, frequency_penalty, etc.)
        for k, v in kwargs.items():
            payload[k] = v
        return payload

    def _post_with_retry(
        self,
        path: str,
        payload: JsonDict,
        max_retries: int = 5,
        base_backoff: float = 0.8,
        jitter: float = 0.2,
    ) -> JsonDict:
        url = self.base_url + path

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                if r.status_code >= 200 and r.status_code < 300:
                    return r.json()

                # retry on common transient codes
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    raise LLMError(
                        "Transient HTTP error",
                        status_code=r.status_code,
                        response_text=r.text,
                    )

                raise LLMError(
                    "HTTP error",
                    status_code=r.status_code,
                    response_text=r.text,
                )

            except (requests.Timeout, requests.ConnectionError, LLMError) as e:
                last_err = e
                if attempt >= max_retries:
                    break
                sleep_s = base_backoff * (2 ** attempt)
                sleep_s *= 1.0 + random.uniform(-jitter, jitter)
                time.sleep(max(0.0, sleep_s))

        if isinstance(last_err, Exception):
            raise last_err
        raise LLMError("Unknown error in _post_with_retry")

    def _post_stream_sse(
        self,
        path: str,
        payload: JsonDict,
    ) -> Generator[str, None, None]:
        """
        SSE streaming parser:
        - expects lines like: "data: {json}\n"
        - terminates on "data: [DONE]"
        - yields delta content tokens
        """
        url = self.base_url + path
        with self._session.post(url, json=payload, timeout=self.timeout, stream=True) as r:
            if not (200 <= r.status_code < 300):
                raise LLMError(
                    "HTTP error (stream)",
                    status_code=r.status_code,
                    response_text=r.text,
                )

            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break

                # parse chunk json
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    # ignore malformed chunks
                    continue

                # OpenAI-style streaming:
                # chunk["choices"][0]["delta"]["content"]
                try:
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    tok = delta.get("content", "")
                    if tok:
                        yield tok
                except Exception:
                    continue

    def _extract_text(self, data: JsonDict) -> str:
        """
        OpenAI-compatible: choices[0].message.content
        """
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMError("No choices in response", response_text=json.dumps(data)[:5000])
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if content is None:
                content = ""
            return str(content)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Failed to parse response: {e}", response_text=json.dumps(data)[:5000])

    @staticmethod
    def _extract_first_json(text: str) -> Optional[str]:
        """
        Best-effort: extract the first top-level JSON object/array substring.
        """
        s = text.strip()
        if not s:
            return None

        # Find first '{' or '['
        start = None
        for i, ch in enumerate(s):
            if ch in "{[":
                start = i
                break
        if start is None:
            return None

        opener = s[start]
        closer = "}" if opener == "{" else "]"

        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return s[start : j + 1]
        return None


# ---------------------------
# Optional: tiny helper to build Stage B1/B2/B3 prompts
# ---------------------------

def build_stage_b_prompt(
    task: str,
    hpo_id: str,
    hpo_name: str,
    context: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Minimal prompt builder for your Stage B tasks.
    task: "b1" | "b2" | "b3"

    You can replace this with your real pipeline prompt logic.
    """
    extra = extra or {}
    header = {
        "task": task,
        "hpo_id": hpo_id,
        "hpo_name": hpo_name,
        "extra": extra,
    }
    return (
        "You are an information extraction assistant.\n"
        "Return STRICT JSON only.\n\n"
        f"INPUT_META={json.dumps(header, ensure_ascii=False)}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "OUTPUT_SCHEMA:\n"
        "- If task==b1: {\"candidates\":[{\"span\":\"...\",\"reason\":\"...\"}, ...]}\n"
        "- If task==b2: {\"positives\":[\"...\"],\"hard_negatives\":[\"...\"]}\n"
        "- If task==b3: {\"bioes\":[{\"span\":\"...\",\"label\":\"B|I|O|E|S\"}],\"notes\":\"...\"}\n"
    )


if __name__ == "__main__":
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set env DEEPSEEK_API_KEY first.")

    client = LLMClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        timeout=60.0,
    )

    prompt = build_stage_b_prompt(
        task="b1",
        hpo_id="HP:0001250",
        hpo_name="Seizure",
        context="The patient experienced recurrent generalized tonic-clonic seizures over 2 years.",
    )

    obj = client.complete_json(
        system="You extract candidate spans for phenotype mentions.",
        user=prompt,
        temperature=0.0,
        response_format=True,  # set False if your server rejects response_format
    )
    print(json.dumps(obj, ensure_ascii=False, indent=2))
