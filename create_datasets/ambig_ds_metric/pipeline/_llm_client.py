"""Tiny shared helper: OpenAI-compatible chat client.

All scripts in this folder import `make_client()` and `call_llm()` from here so
the API/key handling stays in one place.

Configuration (environment variables):
    OPENAI_API_KEY   required
    OPENAI_BASE_URL  optional, defaults to the official OpenAI endpoint.
                     Override to use any OpenAI-compatible endpoint
                     (vLLM, OpenRouter, Azure, internal gateway, ...).
    AMBIG_LLM_MODEL  optional, defaults to "gpt-4o-mini".
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("AMBIG_LLM_MODEL", "gpt-4o-mini")


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # Optional fallback: a .env file next to this script.
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No API key found. Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL).")


def make_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=load_api_key())


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str,
             model: str = DEFAULT_MODEL, temperature: float | None = None,
             max_attempts: int = 3, max_tokens: int | None = None) -> str:
    """Call the chat completions API with retry. Returns content (or '').

    `temperature` is omitted unless explicitly passed; some gateway models
    (e.g. anthropic_claude_opus_4_7) reject it.

    `max_tokens` defaults to the env var `AMBIG_LLM_MAX_TOKENS` (16384). The
    redaction step needs a high cap because some Kaggle prompts are >30 KB
    and the redacted output is roughly the same length as the input; gateway
    defaults of ~2-4 K tokens silently truncate them.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if max_tokens is None:
        try:
            max_tokens = int(os.environ.get("AMBIG_LLM_MAX_TOKENS", "16384"))
        except ValueError:
            max_tokens = 16384
    kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
    if temperature is not None:
        kwargs["temperature"] = temperature
    last_err = None
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if content:
                return content
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    if last_err:
        print(f"  [llm] giving up after {max_attempts} attempts: {last_err}",
              file=sys.stderr)
    return ""
