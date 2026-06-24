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
