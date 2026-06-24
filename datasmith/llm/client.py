"""LLM client — OpenAI-compatible API (Groq, OpenRouter, Gemini).

Ponytail: raw requests, no SDKs. Works with any OpenAI-compatible endpoint.
Supports Groq, OpenRouter, and Gemini out of the box.
"""

import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Provider presets
_PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-32b:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
    },
}

_DEFAULT_PROVIDER = "groq"


def _get_config() -> tuple[str, str, str, str]:
    """Read LLM config from env vars.

    Priority: GEMINI_API_KEY → GROQ_API_KEY → OPENROUTER_API_KEY → LLM_API_KEY.
    Override endpoint with LLM_BASE_URL and model with LLM_MODEL.
    Returns (api_key, base_url, model, provider_name).
    """
    # Check explicit provider env vars first
    for pname, cfg in _PROVIDERS.items():
        key = os.environ.get(cfg["env_key"])
        if key:
            return (
                key,
                os.environ.get("LLM_BASE_URL", cfg["base_url"]),
                os.environ.get("LLM_MODEL", cfg["model"]),
                pname,
            )

    # Catch-all
    fallback_key = os.environ.get("LLM_API_KEY", "")
    return (
        fallback_key,
        os.environ.get("LLM_BASE_URL", _PROVIDERS[_DEFAULT_PROVIDER]["base_url"]),
        os.environ.get("LLM_MODEL", _PROVIDERS[_DEFAULT_PROVIDER]["model"]),
        _DEFAULT_PROVIDER,
    )


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
    api_key, base_url, model, provider = _get_config()
    if not api_key:
        logger.warning("No LLM API key configured")
        return None

    logger.debug("LLM call: provider=%s model=%s", provider, model)

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
    key, _, _, _ = _get_config()
    return bool(key)
