#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_client.py

Minimal, prompt-agnostic DeepSeek (OpenAI-compatible) chat client.

Design goals:
- NO default system prompt
- NO embedded prompt templates
- Caller must explicitly provide system_prompt and user_prompt
- Supports:
    - run()        : non-streaming, return full text
    - run_stream() : streaming, yield text chunks
    - run_json()   : optional helper for JSON-returning prompts

This client is intentionally "dumb":
- It does NOT know about scale2/3/4
- It does NOT parse lines or enforce constraints
- It does NOT assume JSON unless explicitly requested

All prompt logic lives outside this class.
"""

import os
import json
import requests
from typing import Optional, Generator, Dict, Any, List


class LLMClient:
    """
    Generic DeepSeek API client.

    Parameters
    ----------
    api_key : str
        DeepSeek API key.
    base_url : str
        API base URL (default: https://api.deepseek.com).
    model : str
        Model name (default: deepseek-chat).
    timeout : float
        Request timeout in seconds.
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
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-style messages array.

        system_prompt is optional.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _post(self, payload: Dict[str, Any], stream: bool = False):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """
        Non-streaming call.
        Returns full model output as a single string.
        """

        payload = {
            "model": self.model,
            "messages": self._build_messages(system_prompt, user_prompt),
            "temperature": temperature,
            "stream": False,
        }

        try:
            r = self._post(payload, stream=False)
            r.raise_for_status()
        except Exception as e:
            print(f"❌ LLMClient.run request failed: {e}")
            return ""

        try:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            print("⚠️ Failed to parse response JSON. Raw response:")
            try:
                print(r.text)
            except Exception:
                pass
            return ""

    def run_stream(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """
        Streaming call.
        Yields text chunks as they arrive.
        """

        payload = {
            "model": self.model,
            "messages": self._build_messages(system_prompt, user_prompt),
            "temperature": temperature,
            "stream": True,
        }

        try:
            with self._post(payload, stream=True) as r:
                r.raise_for_status()

                for line in r.iter_lines():
                    if not line:
                        continue

                    # Server-Sent Events format: "data: {...}"
                    if line.startswith(b"data: "):
                        data = line[len(b"data: ") :].decode("utf-8")

                        if data.strip() == "[DONE]":
                            break

                        try:
                            part = json.loads(data)
                            delta = part["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            # Ignore malformed chunks
                            continue

        except Exception as e:
            print(f"❌ LLMClient.run_stream request failed: {e}")
            return

    def run_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Helper for prompts that are EXPECTED to return JSON.
        Attempts strict parsing first, then falls back to substring extraction.
        """

        text = self.run(system_prompt, user_prompt, temperature)

        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
            except Exception:
                pass

        print("⚠️ JSON parsing failed. Returning raw text.")
        return {"raw": text}


# ----------------------------------------------------------------------
# Minimal usage example (no prompt defaults)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY to run this example.")
        raise SystemExit(0)

    client = LLMClient(api_key=api_key)

    system_prompt = "You are a biomedical text processing assistant."
    user_prompt = "output three lowercase phrases about limb muscle atrophy"

    out = client.run(system_prompt, user_prompt)
    print(out)
