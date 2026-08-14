"""Tests for the revalidate stage — cheap fix validation in the LLM critique.

Applies a cost-gated funnel: LLM critique output (expensive) gets a
deterministic revalidation pass (free, numpy/pandas) before fixes are
applied, so hallucinated or destructive fixes are skipped instead of applied.
"""

import pandas as pd

from datasmith.llm.critique import (
    CritiqueFix,
    _validate_fixes,
)


def _df_with_columns(**cols) -> pd.DataFrame:
    """Build a DataFrame from {col: [values]} dict."""
    return pd.DataFrame(cols)


class TestValidateFixes:
    def test_valid_clamp_passes_through(self):
        df = _df_with_columns(price=[10.0, 20.0, 30.0])
        fix = CritiqueFix(column="price", action="clamp", reason="t", clamp_min=5, clamp_max=25)
        valid, skipped = _validate_fixes([fix], df)
        assert valid == [fix]
        assert skipped == []

    def test_inverted_clamp_is_skipped(self):
        df = _df_with_columns(price=[10.0, 20.0, 30.0])
        fix = CritiqueFix(column="price", action="clamp", reason="t", clamp_min=25, clamp_max=5)
        valid, skipped = _validate_fixes([fix], df)
        assert valid == []
        assert len(skipped) == 1
        assert skipped[0]["column"] == "price"
        assert "clamp" in skipped[0]["reason"].lower()

    def test_clamp_on_non_numeric_column_is_skipped(self):
        df = _df_with_columns(status=["active", "inactive", "pending"])
        fix = CritiqueFix(column="status", action="clamp", reason="t", clamp_min=0, clamp_max=10)
        valid, skipped = _validate_fixes([fix], df)
        assert valid == []
        assert len(skipped) == 1

    def test_clamp_on_missing_column_is_skipped(self):
        df = _df_with_columns(price=[1.0, 2.0])
        fix = CritiqueFix(column="nope", action="clamp", reason="t", clamp_min=0, clamp_max=10)
        valid, skipped = _validate_fixes([fix], df)
        assert valid == []
        assert len(skipped) == 1

    def test_retype_that_nullifies_values_is_skipped(self):
        # Mostly non-numeric text → integer coercion destroys >30% of values
        df = _df_with_columns(code=["abc", "xyz", "123", "qrs"])
        fix = CritiqueFix(column="code", action="retype", reason="t", new_type="integer")
        valid, skipped = _validate_fixes([fix], df)
        assert valid == []
        assert len(skipped) == 1
        assert skipped[0]["action"] == "retype"
        assert "nullify" in skipped[0]["reason"].lower()

    def test_retype_on_clean_numeric_column_passes(self):
        df = _df_with_columns(qty=[1, 2, 3, 4])
        fix = CritiqueFix(column="qty", action="retype", reason="t", new_type="integer")
        valid, skipped = _validate_fixes([fix], df)
        assert valid == [fix]
        assert skipped == []

    def test_retype_on_missing_column_is_skipped(self):
        df = _df_with_columns(qty=[1, 2, 3])
        fix = CritiqueFix(column="nope", action="retype", reason="t", new_type="integer")
        valid, skipped = _validate_fixes([fix], df)
        assert valid == []
        assert len(skipped) == 1

    def test_drop_and_rename_not_handled_here(self):
        """drop/rename validation stays in the apply loop (schema-based)."""
        df = _df_with_columns(a=[1, 2], b=[3, 4])
        drop_fix = CritiqueFix(column="b", action="drop", reason="t")
        rename_fix = CritiqueFix(column="a", action="rename", reason="t", new_name="c")
        valid, skipped = _validate_fixes([drop_fix, rename_fix], df)
        assert valid == [drop_fix, rename_fix]
        assert skipped == []

    def test_mixed_batch_keeps_valid_drops_invalid(self):
        df = _df_with_columns(price=[5.0, 15.0], qty=[1, 2])
        bad = CritiqueFix(column="price", action="clamp", reason="t", clamp_min=50, clamp_max=10)
        good = CritiqueFix(column="qty", action="clamp", reason="t", clamp_min=0, clamp_max=5)
        valid, skipped = _validate_fixes([bad, good], df)
        assert valid == [good]
        assert len(skipped) == 1
        assert skipped[0]["column"] == "price"


class TestValidateFixesEmpty:
    def test_empty_fixes_returns_empty(self):
        df = _df_with_columns(a=[1, 2])
        valid, skipped = _validate_fixes([], df)
        assert valid == []
        assert skipped == []

    def test_empty_dataframe_safe(self):
        df = pd.DataFrame({"price": pd.Series(dtype=float)})
        fix = CritiqueFix(column="price", action="clamp", reason="t", clamp_min=0, clamp_max=10)
        valid, _ = _validate_fixes([fix], df)
        # Empty column is numeric-convertible (0 values nullified) → valid
        assert valid == [fix]
