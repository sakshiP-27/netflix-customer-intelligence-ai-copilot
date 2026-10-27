"""One place that talks to the LLM. Every Week 3 capability should call ask_llm."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PROVIDER = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
MODEL = os.getenv("LLM_MODEL")
API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)
BASE_URL = os.getenv("LLM_BASE_URL")


def _require_config() -> None:
    if not MODEL:
        raise RuntimeError("Set LLM_MODEL in .env (see .env.example).")
    if PROVIDER != "ollama" and not API_KEY:
        raise RuntimeError("Set LLM_API_KEY in .env. Do not put the key in code.")


def _rate_limited(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in {429, 500, 503}:
        return True
    text = str(exc).lower()
    return any(
        needle in text
        for needle in ("429", "resource_exhausted", "rate limit", "unavailable", "503")
    )


def _text_from_gemini(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=API_KEY)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_prompt,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt == 3 or not _rate_limited(exc):
                raise
            time.sleep(20 * (attempt + 1))
    raise last_error or RuntimeError("Gemini request failed")


def _text_from_anthropic(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if getattr(block, "text", None))


def _text_from_openai_compatible(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    kwargs = {"api_key": API_KEY or "ollama"}
    if PROVIDER == "ollama":
        kwargs["base_url"] = BASE_URL or "http://localhost:11434/v1"
    elif BASE_URL:
        kwargs["base_url"] = BASE_URL

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def ask_llm(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Send one system + one user message. Temperature stays 0 so answers stay stable."""
    _require_config()
    if PROVIDER in {"gemini", "google"}:
        return _text_from_gemini(system_prompt, user_prompt, max_tokens)
    if PROVIDER == "anthropic":
        return _text_from_anthropic(system_prompt, user_prompt, max_tokens)
    if PROVIDER in {"openai", "ollama"}:
        return _text_from_openai_compatible(system_prompt, user_prompt, max_tokens)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={PROVIDER!r}. Use gemini, anthropic, openai, or ollama."
    )
