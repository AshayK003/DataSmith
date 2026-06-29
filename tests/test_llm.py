"""Tests for LLM → Schema discovery pipeline."""

import tempfile
from unittest.mock import patch

import pytest

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.llm.schemas import ColumnSchema, NLDiscoveryResult
from datasmith.llm.discovery import (
    _cache_key,
    _parse_llm_response,
    _result_to_schema,
    discover_schema,
)


@pytest.fixture
def db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    database = Database(tmp.name)
    yield database
    database.close()


@pytest.fixture
def kg(db):
    return KnowledgeGraph(db)


class TestCacheKey:
    def test_consistent_hashing(self):
        assert _cache_key("e-commerce data") == _cache_key("  E-Commerce Data  ")

    def test_different_inputs(self):
        assert _cache_key("healthcare") != _cache_key("finance")


class TestParseLLMResponse:
    def test_plain_json(self):
        raw = (
            '{"domain": "e-commerce", "domain_description": "test", '
            '"columns": [{"column_name": "id", "data_type": "text", '
            '"description": "ID"}]}'
        )
        result = _parse_llm_response(raw)
        assert result is not None
        assert result.domain == "e-commerce"
        assert len(result.columns) == 1

    def test_markdown_fenced_json(self):
        raw = (
            '```json\n{"domain": "finance", "domain_description": "test", '
            '"columns": [{"column_name": "amount", "data_type": "numeric", '
            '"description": "Amount"}]}\n```'
        )
        result = _parse_llm_response(raw)
        assert result is not None
        assert result.domain == "finance"

    def test_invalid_json(self):
        assert _parse_llm_response("not json") is None

    def test_missing_required_field(self):
        raw = '{"domain": "test", "domain_description": "test"}'
        assert _parse_llm_response(raw) is None


class TestResultToSchema:
    def test_converts_basic_result(self):
        result = NLDiscoveryResult(
            domain="e-commerce",
            domain_description="Online retail",
            columns=[
                ColumnSchema(column_name="price", data_type="numeric",
                             description="Price", distribution_hint="powerlaw",
                             min=0.99, max=500.0),
                ColumnSchema(column_name="name", data_type="text",
                             description="Product name"),
            ],
        )
        schema = _result_to_schema(result)
        assert len(schema) == 2
        assert schema[0]["column_name"] == "price"
        assert schema[0]["data_type"] == "numeric"
        assert schema[0]["distribution_hint"] == "powerlaw"
        assert schema[1]["data_type"] == "text"
        assert "distribution_hint" not in schema[1]


class TestDiscoverSchema:
    def test_kg_hit_on_domain_name(self, kg):
        """Input that matches a known domain should hit KG first."""
        # Use healthcare (seeded domain) to get KG schema
        schema = discover_schema(kg, "healthcare patient data")
        assert schema is not None
        assert len(schema) > 0
        # Should match KG columns for healthcare
        assert any(c["column_name"] in ("age", "patient_id", "lab_result") for c in schema)

    def test_kg_hit_on_domain_startswith(self, kg):
        schema = discover_schema(kg, "e-commerce orders")
        assert schema is not None
        assert len(schema) > 0

    def test_unknown_domain_no_llm(self, kg):
        """Without API key, unknown domains should return None (generic fallback)."""
        with patch("datasmith.llm.discovery.is_available", return_value=False):
            schema = discover_schema(kg, "something completely unknown 42")
        assert schema is None

    def test_llm_extraction_success(self, kg):
        """When LLM is available, unknown domains should extract."""
        mock_result = NLDiscoveryResult(
            domain="gaming",
            domain_description="Video game sales data",
            columns=[
                ColumnSchema(column_name="game", data_type="text", description="Game title"),
                ColumnSchema(column_name="sales", data_type="numeric", description="Units sold",
                             distribution_hint="powerlaw", min=1000, max=100_000_000),
            ],
        )
        with (
            patch("datasmith.llm.discovery.is_available", return_value=True),
            patch("datasmith.llm.discovery._llm_extract", return_value=mock_result),
        ):
            schema = discover_schema(kg, "video game sales dataset")

        assert schema is not None
        assert len(schema) == 2
        assert schema[0]["data_type"] == "text"
        assert schema[1]["distribution_hint"] == "powerlaw"

        # Should also be cached
        with patch("datasmith.llm.discovery.is_available", return_value=False):
            cached = discover_schema(kg, "video game sales dataset")
        assert cached is not None


class TestVerifySchema:
    """Tests for LLM schema verification (critique.py)."""

    def test_parses_valid_json(self):
        from datasmith.llm.critique import _parse_verification_response
        r = _parse_verification_response(
            '{"relevant": true, "issues_found": 0, "summary": "OK", '
            '"missing_columns": [], "extra_columns": [], "type_suggestions": []}'
        )
        assert r is not None
        assert r.relevant is True

    def test_parses_with_issues(self):
        from datasmith.llm.critique import _parse_verification_response
        r = _parse_verification_response(
            '{"relevant": false, "issues_found": 2, "summary": "Missing age", '
            '"missing_columns": ["age"], "extra_columns": ["x"], '
            '"type_suggestions": ["age should be integer"]}'
        )
        assert r is not None
        assert r.relevant is False
        assert "age" in r.missing_columns

    def test_parses_markdown_fences(self):
        from datasmith.llm.critique import _parse_verification_response
        r = _parse_verification_response(
            '```json\n{"relevant": true, "issues_found": 0, "summary": "OK", '
            '"missing_columns": [], "extra_columns": [], "type_suggestions": []}\n```'
        )
        assert r is not None
        assert r.relevant is True

    def test_parses_extra_commentary(self):
        from datasmith.llm.critique import _parse_verification_response
        r = _parse_verification_response(
            'Here:\n{"relevant": false, "issues_found": 1, "summary": "n/a", '
            '"missing_columns": [], "extra_columns": [], "type_suggestions": []}\nDone.'
        )
        assert r is not None
        assert r.issues_found == 1

    def test_invalid_json_returns_none(self):
        from datasmith.llm.critique import _parse_verification_response
        assert _parse_verification_response("not json") is None
        assert _parse_verification_response("") is None

    def test_returns_none_without_llm(self):
        from datasmith.llm.critique import verify_schema
        result = verify_schema(
            schema=[{"column_name": "x", "data_type": "text"}],
            user_prompt="test",
        )
        assert result is None  # No LLM configured in test env

    def test_returns_none_with_empty_prompt(self):
        from datasmith.llm.critique import verify_schema
        result = verify_schema(
            schema=[{"column_name": "x", "data_type": "text"}],
            user_prompt="",
        )
        assert result is None
