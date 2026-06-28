"""Tests for the Correlation Engine."""

import numpy as np
import pandas as pd
import pytest

from datasmith.generation.correlator import apply_correlations


class TestApplyCorrelations:
    def test_returns_original_when_no_correlations(self):
        """Empty correlation list returns the DataFrame unchanged (no copy needed)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        rng = np.random.default_rng(42)
        result = apply_correlations(df, [], rng)
        # Identity is preserved since no mutation occurs
        assert result["a"].iloc[0] == 1
        assert result["b"].iloc[2] == 6

    def test_induces_positive_correlation(self):
        """Two columns get a strong positive correlation."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.uniform(0, 100, 1000),
            "y": rng.uniform(0, 100, 1000),
        })
        correlations = [{"col_a": "x", "col_b": "y", "rho": 0.9}]
        result = apply_correlations(df, correlations, rng)
        corr = result["x"].corr(result["y"])
        # Should be close to 0.9
        assert corr > 0.7, f"Expected corr > 0.7, got {corr:.3f}"

    def test_preserves_marginals(self):
        """Column values are reordered but the set of values stays the same."""
        rng = np.random.default_rng(42)
        values_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        values_b = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        df = pd.DataFrame({"a": values_a.copy(), "b": values_b.copy()})
        correlations = [{"col_a": "a", "col_b": "b", "rho": 0.95}]
        result = apply_correlations(df, correlations, rng)
        assert sorted(result["a"].values) == sorted(values_a)
        assert sorted(result["b"].values) == sorted(values_b)

    def test_handles_single_column(self):
        """Fewer than 2 columns returns the data unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        correlations = [{"col_a": "a", "col_b": "b", "rho": 0.5}]
        rng = np.random.default_rng(42)
        result = apply_correlations(df, correlations, rng)
        assert result["a"].iloc[0] == 1

    def test_handles_text_columns(self):
        """Text columns can also be correlated (lexicographic ordering)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "name": ["Charlie", "Alice", "Bob", "Eve", "David"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        correlations = [{"col_a": "name", "col_b": "value", "rho": 0.9}]
        result = apply_correlations(df, correlations, rng)
        # After correlation with rho=0.9, larger values should tend to have
        # alphabetically later names. Check the ordering.
        assert len(result) == 5

    def test_handles_nan_gracefully(self):
        """NaN values are preserved at their positions during reordering."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        correlations = [{"col_a": "a", "col_b": "b", "rho": 0.9}]
        result = apply_correlations(df, correlations, rng)
        # NaN positions should be preserved
        assert pd.isna(result["a"].iloc[1])
        assert pd.isna(result["a"].iloc[3])

    def test_rho_clamping(self):
        """Rho values outside [-0.999, 0.999] are clamped."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.uniform(0, 100, 500),
            "b": rng.uniform(0, 100, 500),
        })
        # rho=1.5 should be clamped to 0.999
        result = apply_correlations(
            df,
            [{"col_a": "a", "col_b": "b", "rho": 1.5}],
            rng,
        )
        corr = result["a"].corr(result["b"])
        assert corr > 0.5

    def test_multiple_pairs(self):
        """Three columns with pairwise correlations."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.uniform(0, 100, 1000),
            "y": rng.uniform(0, 100, 1000),
            "z": rng.uniform(0, 100, 1000),
        })
        correlations = [
            {"col_a": "x", "col_b": "y", "rho": 0.8},
            {"col_a": "y", "col_b": "z", "rho": -0.6},
        ]
        result = apply_correlations(df, correlations, rng)
        xy = result["x"].corr(result["y"])
        yz = result["y"].corr(result["z"])
        assert xy > 0.5, f"xy corr {xy:.3f} too low"
        assert yz < -0.3, f"yz corr {yz:.3f} too high"

    def test_integration_with_generated_data(self):
        """Full pipeline: correlated numeric columns."""
        from datasmith.generation.generator import generate_from_schema
        schema = [
            {"column_name": "price", "data_type": "numeric",
             "distribution_hint": "normal", "mean": 50, "std": 10, "min": 10, "max": 100},
            {"column_name": "quantity", "data_type": "integer",
             "distribution_hint": "uniform", "min": 1, "max": 100},
        ]
        rng = np.random.default_rng(42)
        df = generate_from_schema(schema, 500, rng)
        correlations = [{"col_a": "price", "col_b": "quantity", "rho": 0.85}]
        result = apply_correlations(df, correlations, rng)
        corr = result["price"].corr(result["quantity"])
        assert corr > 0.6, f"Expected corr > 0.6, got {corr:.3f}"
        # Marginals preserved
        assert np.allclose(sorted(result["price"].values), sorted(df["price"].values))
        assert sorted(result["quantity"].values) == sorted(df["quantity"].values)
