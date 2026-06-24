"""Tests for edge cases in the generation pipeline."""

import numpy as np
import pandas as pd
import pytest

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.generation.engine import generate_dataset
from datasmith.generation.generator import generate_column


class TestGenerateDatasetEmptyCustomSchema:
    """Empty custom_schema list falls back to generic schema."""

    def test_generate_dataset_empty_custom_schema(self):
        db = Database(":memory:")
        try:
            kg = KnowledgeGraph(db)
            df = generate_dataset(kg, "e-commerce", custom_schema=[], seed=42)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 100
            # Generic e-commerce schema includes these columns
            assert "order_id" in df.columns
            assert "price" in df.columns
            assert "quantity" in df.columns
        finally:
            db.close()


class TestGenerateColumnPowerlawDegenerate:
    """Power-law sampler with near-zero variance returns finite values."""

    def test_generate_column_powerlaw_degenerate(self):
        rng = np.random.default_rng(42)
        stats = {"distribution_hint": "powerlaw", "mean": 0.0,
                 "std": 0.001, "min": 0, "max": 1}
        data = generate_column("test", "numeric", stats, 100, rng)
        assert len(data) == 100
        assert np.all(np.isfinite(data))


class TestGenerateColumnLognormalZeroMean:
    """Lognormal sampler with zero mean returns finite values."""

    def test_generate_column_lognormal_zero_mean(self):
        rng = np.random.default_rng(42)
        stats = {"distribution_hint": "lognormal", "mean": 0,
                 "std": 1, "min": 0, "max": 100}
        data = generate_column("test", "numeric", stats, 100, rng)
        assert len(data) == 100
        assert np.all(np.isfinite(data))


class TestGenerateColumnUnknownDistribution:
    """Unknown distribution hint falls back gracefully."""

    def test_generate_column_unknown_distribution(self):
        rng = np.random.default_rng(42)
        stats = {"distribution_hint": "exponential"}
        data = generate_column("test", "numeric", stats, 100, rng)
        assert len(data) == 100
        assert np.all(np.isfinite(data))
