"""Tests for the Schema Enricher module."""

import numpy as np

from datasmith.schema.enricher import enrich_schema


class TestEnrichSchema:
    def test_enriches_year_to_integer(self):
        """Year columns get data_type='integer' with range."""
        cols = [{"column_name": "year", "data_type": "numeric", "description": "Year of purchase"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "integer"
        assert result[0]["min"] == 2015
        assert result[0]["max"] == 2024
        assert result[0]["distribution_hint"] == "uniform"

    def test_enriches_age_to_integer(self):
        """Age columns get data_type='integer' with range."""
        cols = [{"column_name": "age", "data_type": "numeric"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "integer"
        assert result[0]["min"] == 18
        assert result[0]["max"] == 90

    def test_enriches_email_description(self):
        """Email columns get proper description for text_profiles matching."""
        cols = [{"column_name": "email", "data_type": "text"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "text"
        assert "email" in result[0].get("description", "").lower()

    def test_leaves_explicit_values_unchanged(self):
        """Explicitly set non-data_type values are preserved."""
        cols = [{"column_name": "age", "data_type": "numeric",
                 "min": 0, "max": 120}]
        result = enrich_schema(cols)
        # data_type gets overridden to integer
        assert result[0]["data_type"] == "integer"
        # But explicit min/max are preserved
        assert result[0]["min"] == 0
        assert result[0]["max"] == 120

    def test_handles_multiple_columns(self):
        """Mixed column types all get enriched correctly."""
        cols = [
            {"column_name": "year", "data_type": "numeric"},
            {"column_name": "price", "data_type": "numeric"},
            {"column_name": "country", "data_type": "text"},
            {"column_name": "rating", "data_type": "numeric"},
        ]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "integer"
        assert result[1]["data_type"] == "numeric"
        assert result[1].get("distribution_hint") == "powerlaw"
        assert result[2]["data_type"] == "text"
        assert result[3].get("min") == 1
        assert result[3].get("max") == 5

    def test_unknown_column_unchanged(self):
        """Columns with no semantic match get sane defaults but no special handling."""
        cols = [{"column_name": "weird_field", "data_type": "text"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "text"
        assert result[0]["column_name"] == "weird_field"

    def test_handles_unknown_numeric_column(self):
        """Numeric columns without semantic match get generic defaults."""
        cols = [{"column_name": "custom_field", "data_type": "numeric"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "numeric"
        # Falls through to _fill_range with normal distribution
        assert result[0].get("min") == 0
        assert result[0].get("max") == 100

    def test_score_maps_to_rating_semantics(self):
        """Column named 'score' gets rating constraints (1-5, normal)."""
        cols = [{"column_name": "score", "data_type": "numeric"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "numeric"
        assert result[0]["min"] == 1
        assert result[0]["max"] == 5
        assert result[0].get("distribution_hint") == "normal"

    def test_fills_missing_range_for_numeric(self):
        """Numeric columns without min/max get defaults from distribution."""
        cols = [{"column_name": "measurement", "data_type": "numeric",
                 "distribution_hint": "uniform"}]
        result = enrich_schema(cols)
        assert result[0].get("min") == 0
        assert result[0].get("max") == 100

    def test_enriches_price_with_powerlaw(self):
        """Price columns get powerlaw distribution by default."""
        cols = [{"column_name": "price", "data_type": "numeric"}]
        result = enrich_schema(cols)
        assert result[0]["distribution_hint"] == "powerlaw"
        assert result[0]["min"] == 0.99
        assert result[0]["max"] == 999.99

    def test_enriches_quantity_to_integer(self):
        """Quantity columns get integer type with powerlaw."""
        cols = [{"column_name": "quantity", "data_type": "numeric"}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "integer"
        assert result[0]["distribution_hint"] == "uniform"  # overridden from powerlaw for integer

    def test_handles_empty_list(self):
        """Empty schema list returns empty list."""
        assert enrich_schema([]) == []

    def test_handles_none_values(self):
        """Columns with None values are handled gracefully."""
        cols = [{"column_name": "year", "data_type": None, "description": None}]
        result = enrich_schema(cols)
        assert result[0]["data_type"] == "integer"

    def test_integration_with_generated_output(self):
        """Enriched schema produces correct dtypes when generated."""
        from datasmith.generation.generator import generate_from_schema
        cols = [
            {"column_name": "year", "data_type": "numeric"},
            {"column_name": "age", "data_type": "numeric"},
            {"column_name": "price", "data_type": "numeric"},
        ]
        enriched = enrich_schema(cols)
        rng = np.random.default_rng(42)
        df = generate_from_schema(enriched, 50, rng)
        assert df["year"].dtype == np.int64
        assert df["age"].dtype == np.int64
        assert df["price"].dtype == np.float64
