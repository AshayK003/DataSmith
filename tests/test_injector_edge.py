"""Tests for imperfection injector edge cases."""

import numpy as np
import pandas as pd
import pytest

from datasmith.imperfections.injector import inject_nulls, inject_outliers


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
