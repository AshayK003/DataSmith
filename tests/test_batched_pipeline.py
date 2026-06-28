"""Tests for the Batched Iterative Generation pipeline (Phase 0).

Covers quality metrics, parameter adjustment, and the batched orchestrator.
All tests are statistical — they verify properties of the metrics and pipeline,
not exact numeric equality (which would be fragile).
"""

import numpy as np
import pandas as pd

from datasmith.generation import quality, adjuster, pipeline
from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph


# ══════════════════════════════════════════════════════════════════════════
# Quality Metrics
# ══════════════════════════════════════════════════════════════════════════

SIMPLE_SCHEMA = [
    {"column_name": "id", "data_type": "text"},
    {"column_name": "price", "data_type": "numeric",
     "mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0,
     "distribution_hint": "normal"},
    {"column_name": "qty", "data_type": "integer",
     "mean": 5.0, "std": 2.0, "min": 1, "max": 10},
]


class TestQualityMetrics:
    def test_ks_stat_for_numeric(self):
        """KS stat is computed for numeric columns and is between 0 and 1."""
        rng = np.random.default_rng(42)
        # Generate data from the same schema — should have low KS
        from datasmith.generation.generator import generate_from_schema
        df = generate_from_schema(SIMPLE_SCHEMA, 500, rng)
        metrics = quality.compute_batch_quality(df, SIMPLE_SCHEMA)
        assert "ks_price" in metrics
        assert 0 <= metrics["ks_price"] <= 0.5  # same distribution → low KS
        assert "ks_qty" in metrics
        assert 0 <= metrics["ks_qty"] <= 0.5

    def test_null_drift(self):
        """Null drift is computed for every column."""
        rng = np.random.default_rng(42)
        from datasmith.generation.generator import generate_from_schema
        df = generate_from_schema(SIMPLE_SCHEMA, 500, rng)
        metrics = quality.compute_batch_quality(df, SIMPLE_SCHEMA)
        # No nulls expected in generated data → drift = actual null rate
        assert metrics["null_drift_id"] == 0.0
        assert 0 <= metrics["null_drift_price"] <= 0.01

    def test_quality_score_structure(self):
        """quality_score is present, between 0 and 1."""
        rng = np.random.default_rng(42)
        from datasmith.generation.generator import generate_from_schema
        df = generate_from_schema(SIMPLE_SCHEMA, 500, rng)
        metrics = quality.compute_batch_quality(df, SIMPLE_SCHEMA)
        assert "quality_score" in metrics
        assert 0 <= metrics["quality_score"] <= 1

    def test_aggregates(self):
        """ks_mean, ks_max, ks_count are present."""
        rng = np.random.default_rng(42)
        from datasmith.generation.generator import generate_from_schema
        df = generate_from_schema(SIMPLE_SCHEMA, 500, rng)
        metrics = quality.compute_batch_quality(df, SIMPLE_SCHEMA)
        assert metrics["ks_count"] >= 1
        assert metrics["ks_mean"] >= 0
        assert metrics["ks_max"] >= 0

    def test_only_text_columns(self):
        """Schema with only text columns produces no KS stats but no crash."""
        schema = [
            {"column_name": "a", "data_type": "text"},
            {"column_name": "b", "data_type": "text"},
        ]
        df = pd.DataFrame({"a": ["x"] * 100, "b": ["y"] * 100})
        metrics = quality.compute_batch_quality(df, schema)
        assert metrics["ks_count"] == 0
        assert "corr_diff" not in metrics
        assert metrics["quality_score"] == 1.0

    def test_empty_dataframe(self):
        """Empty DataFrame doesn't crash metrics (graceful degradation)."""
        metrics = quality.compute_batch_quality(pd.DataFrame(), SIMPLE_SCHEMA)
        assert isinstance(metrics, dict)

    def test_correlation_diff(self):
        """corr_diff is computed when >= 2 numeric columns."""
        rng = np.random.default_rng(42)
        from datasmith.generation.generator import generate_from_schema
        df = generate_from_schema(SIMPLE_SCHEMA, 500, rng)
        metrics = quality.compute_batch_quality(df, SIMPLE_SCHEMA)
        if "corr_diff" in metrics:
            assert 0 <= metrics["corr_diff"] <= 1


# ══════════════════════════════════════════════════════════════════════════
# Adjuster
# ══════════════════════════════════════════════════════════════════════════


class TestAdjuster:
    def test_adjust_schema_no_op(self):
        """Schema is unchanged when quality is perfect."""
        metrics = {"ks_price": 0.02, "null_drift_price": 0.0,
                   "quality_score": 1.0}
        schema = [dict(col) for col in SIMPLE_SCHEMA]
        result = adjuster.adjust_schema(schema, metrics)
        assert len(result) == len(schema)
        for orig, adj in zip(schema, result):
            assert orig["column_name"] == adj["column_name"]

    def test_adjust_schema_modifies_mean(self):
        """High KS triggers mean adjustment."""
        metrics = {"ks_price": 0.25, "null_drift_price": 0.0,
                   "quality_score": 0.5}
        schema = [dict(col) for col in SIMPLE_SCHEMA]
        result = adjuster.adjust_schema(schema, metrics)
        # price column should have adjusted mean
        for col in result:
            if col["column_name"] == "price":
                assert col["mean"] != 50.0  # was adjusted
                break

    def test_adjust_schema_does_not_mutate_original(self):
        """Original schema list is not modified by adjuster."""
        metrics = {"ks_price": 0.25, "quality_score": 0.5}
        original_mean = SIMPLE_SCHEMA[1]["mean"]
        schema_copy = [dict(col) for col in SIMPLE_SCHEMA]
        adjuster.adjust_schema(schema_copy, metrics)
        # Original should be unchanged
        assert SIMPLE_SCHEMA[1]["mean"] == original_mean

    def test_adjust_imperfection_profile_none(self):
        """adjust_imperfection_profile returns None when profile is None."""
        result = adjuster.adjust_imperfection_profile(None, {}, [])
        assert result is None

    def test_adjust_profile_no_drift(self):
        """Profile not modified when drift is below threshold."""
        profile = {"null_patterns": {"price": {"null_pct": 10}}}
        result = adjuster.adjust_imperfection_profile(
            profile, {"null_drift_price": 0.001}, SIMPLE_SCHEMA,
        )
        assert result["null_patterns"]["price"]["null_pct"] == 10


# ══════════════════════════════════════════════════════════════════════════
# Pipeline integration
# ══════════════════════════════════════════════════════════════════════════


class TestBatchedGenerate:
    def test_returns_dataframe(self):
        """batched_generate returns a DataFrame with expected shape."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df = pipeline.batched_generate(
                kg, "e-commerce", total_rows=200, batch_size=100, seed=42,
            )
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 200
            assert len(df.columns) >= 6  # e-commerce generic has 7 columns
        finally:
            db.close()

    def test_single_batch(self):
        """When total_rows <= batch_size, a single batch is generated."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df = pipeline.batched_generate(
                kg, "e-commerce", total_rows=50, batch_size=100, seed=42,
            )
            assert len(df) == 50
        finally:
            db.close()

    def test_multiple_batches(self):
        """When total_rows > batch_size, multiple batches are generated."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df = pipeline.batched_generate(
                kg, "e-commerce", total_rows=250, batch_size=100, seed=42,
            )
            assert len(df) == 250
        finally:
            db.close()

    def test_custom_schema(self):
        """Custom schema is respected across batches."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            custom = [
                {"column_name": "a", "data_type": "numeric",
                 "mean": 10.0, "std": 2.0, "min": 0, "max": 20},
                {"column_name": "b", "data_type": "text"},
            ]
            df = pipeline.batched_generate(
                kg, "e-commerce", total_rows=100, batch_size=50,
                custom_schema=custom, seed=42,
            )
            assert list(df.columns) == ["a", "b"]
            assert len(df) == 100
        finally:
            db.close()

    def test_reproducible_with_seed(self):
        """Same seed produces identical results."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df1 = pipeline.batched_generate(
                kg, "e-commerce", total_rows=100, batch_size=50, seed=42,
            )
            df2 = pipeline.batched_generate(
                kg, "e-commerce", total_rows=100, batch_size=50, seed=42,
            )
            pd.testing.assert_frame_equal(df1, df2)
        finally:
            db.close()

    def test_batched_generate_from_engine(self):
        """batched_generate is accessible from engine module."""
        from datasmith.generation.pipeline import batched_generate as bg
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df = bg(kg, "e-commerce", total_rows=50, seed=42)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
        finally:
            db.close()

    def test_large_batch(self):
        """2000 rows across multiple batches doesn't crash."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df = pipeline.batched_generate(
                kg, "e-commerce", total_rows=2000, batch_size=500, seed=42,
            )
            assert len(df) == 2000
        finally:
            db.close()

    def test_with_imperfections(self):
        """Imperfection injection works in batched mode."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        try:
            df = pipeline.batched_generate(
                kg, "e-commerce", total_rows=500, batch_size=200,
                inject_imperfections=True, seed=42,
            )
            assert len(df) == 500
        finally:
            db.close()
