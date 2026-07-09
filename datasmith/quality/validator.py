"""Quality Validator — validates generated DataFrames against column schema.

Runs a battery of quality checks on generated data to catch:
- Integer columns that somehow produced floats
- Values outside declared bounds
- Missing/null integrity issues
- Format violations (emails without @, etc.)

The engine uses this to auto-retry generation with a different seed
when output fails quality gates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────────────────────


@dataclass
class ValidationError:
    """A single quality violation found in generated data."""

    column: str
    check: str  # e.g. "integer_check", "bounds_check", "null_check", "format_check"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating a DataFrame against its column schema."""

    passed: bool
    errors: list[ValidationError] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def __len__(self) -> int:
        return len(self.errors)

    @property
    def summary(self) -> str:
        if self.passed:
            return "✅ All quality checks passed"
        lines = [f"❌ {len(self.errors)} quality failure(s):"]
        for err in self.errors[:5]:
            lines.append(f"  - [{err.check}] {err.column}: {err.message}")
        if len(self.errors) > 5:
            lines.append(f"  ️  ... and {len(self.errors) - 5} more")
        return "\n".join(lines)


# ── Checks ──────────────────────────────────────────────────────────────────


def check_integers(df: pd.DataFrame, schema: list[dict]) -> list[ValidationError]:
    """Check that all integer-typed columns contain no fractional values."""
    errors: list[ValidationError] = []
    for col in schema:
        name = col["column_name"]
        if col.get("data_type") != "integer" or name not in df.columns:
            continue

        series = df[name]
        non_null = series.dropna()

        if len(non_null) == 0:
            continue  # all-null columns will be caught elsewhere

        # Check dtype is int64
        if series.dtype != np.int64:
            # Check actual values have no fractional part
            try:
                is_float = np.issubdtype(series.dtype, np.floating)
            except TypeError:
                is_float = False  # extension dtype (Int64Dtype, etc.)
            if is_float:
                frac = non_null % 1
                bad = (frac != 0).sum()
                if bad > 0:
                    fraction = float(bad / len(non_null))
                    errors.append(ValidationError(
                        column=name,
                        check="integer_check",
                        message=(f"{bad}/{len(non_null)} values "
                                 f"({fraction:.1%}) have fractional parts"),
                        details={"bad_count": int(bad), "total": len(non_null)},
                    ))

    return errors


def check_bounds(df: pd.DataFrame, schema: list[dict]) -> list[ValidationError]:
    """Check that all values with min/max bounds stay within them."""
    errors: list[ValidationError] = []
    for col in schema:
        name = col["column_name"]
        lo = col.get("min")
        hi = col.get("max")
        if lo is None and hi is None:
            continue
        if name not in df.columns:
            continue

        series = df[name]
        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        try:
            vals = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(vals) == 0:
                continue
            if lo is not None and (vals < lo).any():
                bad = int((vals < lo).sum())
                errors.append(ValidationError(
                    column=name,
                    check="bounds_check",
                    message=f"{bad} values below min={lo}",
                    details={"below": bad, "min": lo},
                ))
            if hi is not None and (vals > hi).any():
                bad = int((vals > hi).sum())
                errors.append(ValidationError(
                    column=name,
                    check="bounds_check",
                    message=f"{bad} values above max={hi}",
                    details={"above": bad, "max": hi},
                ))
        except (TypeError, ValueError):
            pass

    return errors


def check_nulls(df: pd.DataFrame, schema: list[dict]) -> list[ValidationError]:
    """Flag columns that are entirely null or have suspiciously high null rates.

    Note: the imperfection injector intentionally introduces nulls, so this
    only flags columns that are 100% null (which would break downstream use).
    """
    errors: list[ValidationError] = []
    for col in schema:
        name = col["column_name"]
        if name not in df.columns:
            continue

        series = df[name]
        null_count = int(series.isnull().sum())
        total = len(series)

        if null_count == total:
            errors.append(ValidationError(
                column=name,
                check="null_check",
                message=f"Column is entirely null ({total}/{total} rows)",
                details={"null_count": null_count, "total": total},
            ))

    return errors


def check_formats(df: pd.DataFrame, schema: list[dict]) -> list[ValidationError]:
    """Check format integrity for known column types (email, phone, ID)."""
    errors: list[ValidationError] = []

    for col in schema:
        name = col["column_name"]
        if name not in df.columns:
            continue

        desc = (col.get("description") or "").lower()
        series = df[name]
        non_null = series.dropna().astype(str)
        # Drop string representations of null to avoid false positives
        non_null = non_null[~non_null.isin(["<NA>", "nan", "NaN", "None"])]
        if len(non_null) == 0:
            continue

        # Email check — description contains "email"
        if re.search(r"email", desc):
            no_at = (~non_null.str.contains("@", regex=False)).sum()
            if no_at > 0:
                errors.append(ValidationError(
                    column=name,
                    check="format_check",
                    message=f"{no_at}/{len(non_null)} values missing '@'",
                    details={"bad_count": int(no_at), "total": len(non_null)},
                ))

        # Phone check — description contains "phone"
        if re.search(r"phone", desc):
            no_digits = (non_null.str.count(r"\d") < 5).sum()
            if no_digits > 0:
                errors.append(ValidationError(
                    column=name,
                    check="format_check",
                    message=f"{no_digits}/{len(non_null)} values have fewer than 5 digits",
                    details={"bad_count": int(no_digits), "total": len(non_null)},
                ))

    
        # ID format check — description contains "id" or column name matches *_id
        if re.search(r"\bid\b", desc) or re.search(r"_id$", name, re.I):
            bad_id = non_null[~non_null.str.match(r"^[A-Z]+-\d+$")]
            if len(bad_id) > 0:
                errors.append(ValidationError(
                    column=name,
                    check="format_check",
                    message=f"{len(bad_id)}/{len(non_null)} values don't match ID format (PREFIX-NUMBERS)",
                    details={"bad_count": int(len(bad_id)), "total": len(non_null)},
                ))

    return errors


def check_diversity(df: pd.DataFrame, schema: list[dict]) -> list[ValidationError]:
    """Check text columns have at least some diversity (not all identical)."""
    errors: list[ValidationError] = []
    for col in schema:
        name = col["column_name"]
        dtype = col.get("data_type", "text")
        if dtype not in ("text", "string") or name not in df.columns:
            continue

        series = df[name]
        non_null = series.dropna()
        if len(non_null) < 2:
            continue

        unique_count = int(non_null.nunique())
        # Flag if every value is the same
        if unique_count <= 1:
            errors.append(ValidationError(
                column=name,
                check="diversity_check",
                message=(f"Column has {unique_count} unique value(s) "
                         f"across {len(non_null)} non-null rows"
                         " — likely a generation failure"),
                details={"unique": unique_count, "non_null": len(non_null)},
            ))

    return errors


# ── Main entry point ────────────────────────────────────────────────────────


def validate(df: pd.DataFrame, schema: list[dict]) -> ValidationResult:
    """Run all quality checks on a generated DataFrame.

    Args:
        df: The generated DataFrame (post-imperfection injection).
        schema: The column schema list used to generate the data.

    Returns:
        ValidationResult with passed flag and detailed errors.
    """
    all_errors: list[ValidationError] = []

    all_errors.extend(check_integers(df, schema))
    all_errors.extend(check_bounds(df, schema))
    all_errors.extend(check_nulls(df, schema))
    all_errors.extend(check_formats(df, schema))
    all_errors.extend(check_diversity(df, schema))

    return ValidationResult(
        passed=len(all_errors) == 0,
        errors=all_errors,
    )
