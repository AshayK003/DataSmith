"""LLM client — OpenAI-compatible API (Groq, OpenRouter, etc.).

Ponytail: raw requests, no SDKs. Works with any OpenAI-compatible endpoint.
"""

import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default: Groq (free, fast)
_DEFAULT_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get_config() -> tuple[str, str, str]:
    """Read LLM config from env vars. Falls back to Groq free tier."""
    api_key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )
    base_url = os.environ.get("LLM_BASE_URL", _DEFAULT_BASE)
    model = os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
    return api_key, base_url, model


def chat_complete(
    system_prompt: str,
    user_prompt: str,
    response_format: Optional[dict] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Call an OpenAI-compatible chat completions endpoint.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User message content.
        response_format: Optional JSON schema dict (like OpenAI structured outputs).
        temperature: Sampling temperature (0.1 for deterministic extraction).
        max_tokens: Max tokens in response.

    Returns:
        Response text, or None on failure.
    """
    api_key, base_url, model = _get_config()
    if not api_key:
        logger.warning("No LLM API key configured")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        body["response_format"] = response_format

    try:
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except requests.Timeout:
        logger.warning("LLM request timed out after 30s")
        return None
    except requests.RequestException as e:
        logger.warning("LLM request failed: %s", e)
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("LLM response parse failed: %s", e)
        return None


def is_available() -> bool:
    """Check if an LLM API key is configured."""
    key, _, _ = _get_config()
    return bool(key)
