"""Tests for LLM client (directly, not via discovery mocks)."""

import requests
from unittest.mock import MagicMock, patch

from datasmith.llm.client import get_config, chat_complete, _PROVIDERS, _DEFAULT_PROVIDER


class TestGetConfigNoEnvVars:
    """get_config returns empty key, default URL/model, groq provider."""

    def test_get_config_no_env_vars(self, monkeypatch):
        for key in ["GROQ_API_KEY", "OPENROUTER_API_KEY",
                     "GEMINI_API_KEY", "LLM_API_KEY",
                     "LLM_BASE_URL", "LLM_MODEL"]:
            monkeypatch.delenv(key, raising=False)

        api_key, base_url, model, provider = get_config()
        assert api_key == ""
        assert base_url == _PROVIDERS[_DEFAULT_PROVIDER]["base_url"]
        assert model == _PROVIDERS[_DEFAULT_PROVIDER]["model"]
        assert provider == _DEFAULT_PROVIDER


class TestChatCompleteHttp403:
    """HTTP 403 returns None."""

    @patch("datasmith.llm.client.requests.post")
    def test_chat_complete_http_403(self, mock_post, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.HTTPError("403 Forbidden")
        )
        mock_post.return_value = mock_response

        result = chat_complete("system", "user")
        assert result is None


class TestChatCompleteMalformedResponse:
    """Missing 'choices' key in response returns None."""

    @patch("datasmith.llm.client.requests.post")
    def test_chat_complete_malformed_response(self, mock_post, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        result = chat_complete("system", "user")
        assert result is None


class TestChatCompleteTimeout:
    """Network timeout returns None."""

    @patch("datasmith.llm.client.requests.post")
    def test_chat_complete_timeout(self, mock_post, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_post.side_effect = requests.Timeout("timed out")

        result = chat_complete("system", "user")
        assert result is None
