"""Tests for imperfection injector edge cases."""

import numpy as np
import pandas as pd
import pytest

from datasmith.imperfections.injector import inject_nulls, inject_outliers, inject_noise


class TestInjectNullsProfileColumnMissingFromDf:
    """inject_nulls skips columns not in the DataFrame."""

    def test_inject_nulls_profile_column_missing_from_df(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df_original = df.copy()
        profile = {
            "null_patterns": {
                "nonexistent": {"null_pct": 50, "pattern": "MCAR"}
            }
        }
        inject_nulls(df, profile, np.random.default_rng(42))
        pd.testing.assert_frame_equal(df, df_original)


class TestInjectNullsIntegerColumnTypeChange:
    """Integer column becomes float64 after null injection."""

    def test_inject_nulls_integer_column_type_change(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        assert df["a"].dtype == np.int64
        profile = {
            "null_patterns": {
                "a": {"null_pct": 50, "pattern": "MCAR"}
            }
        }
        inject_nulls(df, profile, np.random.default_rng(42))
        assert df["a"].dtype == np.float64
        assert df["a"].isna().sum() > 0


class TestInjectOutliersConstantColumn:
    """inject_outliers handles constant columns without crashing."""

    def test_inject_outliers_constant_column(self):
        df = pd.DataFrame({"a": [5.0] * 10})
        profile = {
            "outlier_patterns": {
                "a": {"outlier_pct": 20, "direction": "both"}
            }
        }
        # Should not raise any exception
        inject_outliers(df, profile, np.random.default_rng(42))


class TestInjectNullsMarSelfReference:
    """MAR with self-referencing null_correlations doesn't crash."""

    def test_inject_nulls_mar_self_referencing_cols(self):
        rng = np.random.default_rng(42)
        for cols in [["x", "x"], ["x"]]:
            df = pd.DataFrame({"x": [1.0] * 50})
            profile = {
                "null_patterns": {"x": {"null_pct": 50, "pattern": "MAR"}},
                "null_correlations": [{"cols": cols, "jaccard": 1.0}],
            }
            # Should not raise IndexError
            inject_nulls(df, profile, rng)
            assert df["x"].isna().sum() > 0


class TestInjectNoiseAllNaN:
    """inject_noise on an all-NaN column doesn't crash."""

    def test_inject_noise_all_nan_column(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": [np.nan] * 50})
        profile = {
            "noise_patterns": {"x": {"rounding_pct": 50, "precision": 0.1}},
        }
        # Should not raise ValueError on rng.choice(0, 1)
        inject_noise(df, profile, rng)
        assert df["x"].isna().all()

    def test_inject_noise_zero_rounding_pct(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": [1.234, 5.678, 9.101] * 10})
        before = df["x"].copy()
        profile = {
            "noise_patterns": {"x": {"rounding_pct": 0, "precision": 1.0}},
        }
        # Should not round any values
        inject_noise(df, profile, rng)
        pd.testing.assert_series_equal(before, df["x"])
