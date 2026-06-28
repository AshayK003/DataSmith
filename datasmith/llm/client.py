"""LLM client — OpenAI-compatible API (Groq, OpenRouter, Gemini).

Raw requests, no SDKs. Works with any OpenAI-compatible endpoint.
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

_DEFAULT_PROVIDER = "gemini"


def get_config(
    override_key: str = "",
    override_base_url: str = "",
    override_model: str = "",
) -> tuple[str, str, str, str]:
    """Read LLM config from env vars or explicit overrides.

    Override params take precedence over env vars. Useful for
    user-provided keys in the frontend.

    Priority for key: override_key → env keys → fallback.
    Priority for base_url: override_base_url → LLM_BASE_URL → provider default.
    Priority for model: override_model → LLM_MODEL → provider default.
    Returns (api_key, base_url, model, provider_name).
    """
    if override_key:
        fallback_base = os.environ.get(
            "LLM_BASE_URL", _PROVIDERS[_DEFAULT_PROVIDER]["base_url"]
        )
        fallback_model = os.environ.get(
            "LLM_MODEL", _PROVIDERS[_DEFAULT_PROVIDER]["model"]
        )
        return (
            override_key,
            override_base_url or fallback_base,
            override_model or fallback_model,
            "custom",
        )

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
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Optional[str]:
    """Call an OpenAI-compatible chat completions endpoint.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User message content.
        response_format: Optional JSON schema dict (like OpenAI structured outputs).
        temperature: Sampling temperature (0.1 for deterministic extraction).
        max_tokens: Max tokens in response.
        api_key, base_url, model: Optional overrides (user-provided config).

    Returns:
        Response text, or None on failure.
    """
    effective_key, effective_base, effective_model, provider = get_config(
        override_key=api_key,
        override_base_url=base_url,
        override_model=model,
    )
    if not effective_key:
        logger.warning("No LLM API key configured")
        return None

    logger.debug("LLM call: provider=%s model=%s", provider, model)

    headers = {
        "Authorization": f"Bearer {effective_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": effective_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        body["response_format"] = response_format

    def _do_request(body_overrides: dict | None = None) -> str | None:
        """Inner request helper for retry logic."""
        req_body = dict(body)
        if body_overrides:
            req_body.update(body_overrides)
        try:
            r = requests.post(
                f"{effective_base}/chat/completions",
                headers=headers,
                json=req_body,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.Timeout:
            logger.warning("LLM request timed out after 30s")
            return None
        except requests.RequestException as e:
            logger.warning("LLM request failed: %s", e)
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning("LLM response parse failed: %s", e)
            return None

    # Primary attempt
    content = _do_request()
    if content is not None:
        return content

    # Retry: if response_format was set, some providers (Gemini-compatible)
    # don't support it. Drop it and try again.
    if response_format:
        logger.info("Retrying LLM call without response_format")
        content = _do_request({"response_format": None})
        if content is not None:
            return content

    return None


def is_available() -> bool:
    """Check if an LLM API key is configured."""
    key, _, _, _ = get_config()
    return bool(key)
