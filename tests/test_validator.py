"""Tests for the Quality Validator module."""

import numpy as np
import pandas as pd

from datasmith.quality.validator import (
    ValidationError,
    ValidationResult,
    check_integers,
    check_bounds,
    check_nulls,
    check_formats,
    check_diversity,
    validate,
)


class TestValidationDataModel:
    def test_validation_result_passed(self):
        r = ValidationResult(passed=True)
        assert r
        assert len(r) == 0
        assert "passed" in r.summary

    def test_validation_result_failed(self):
        err = ValidationError(column="x", check="test", message="bad")
        r = ValidationResult(passed=False, errors=[err])
        assert not r
        assert len(r) == 1
        assert "1 quality failure" in r.summary


class TestCheckIntegers:
    def test_passes_clean_integers(self):
        df = pd.DataFrame({"age": np.array([25, 30, 45], dtype=np.int64)})
        schema = [{"column_name": "age", "data_type": "integer"}]
        errors = check_integers(df, schema)
        assert len(errors) == 0

    def test_fails_float_with_fraction(self):
        df = pd.DataFrame({"age": np.array([25.5, 30.1, 45.7])})
        schema = [{"column_name": "age", "data_type": "integer"}]
        errors = check_integers(df, schema)
        assert len(errors) == 1
        assert errors[0].check == "integer_check"

    def test_skips_non_integer_columns(self):
        df = pd.DataFrame({"price": np.array([1.99, 2.50])})
        schema = [{"column_name": "price", "data_type": "numeric"}]
        errors = check_integers(df, schema)
        assert len(errors) == 0

    def test_handles_nulls(self):
        df = pd.DataFrame({"age": pd.array([25, None, 45], dtype=pd.Int64Dtype())})
        schema = [{"column_name": "age", "data_type": "integer"}]
        # Int64 extension dtype — skip since it's not np.float64
        errors = check_integers(df, schema)
        assert len(errors) == 0


class TestCheckBounds:
    def test_passes_within_bounds(self):
        df = pd.DataFrame({"price": np.array([10.0, 50.0, 100.0])})
        schema = [{"column_name": "price", "min": 0, "max": 100}]
        errors = check_bounds(df, schema)
        assert len(errors) == 0

    def test_fails_below_min(self):
        df = pd.DataFrame({"price": np.array([-5.0, 50.0, 100.0])})
        schema = [{"column_name": "price", "min": 0, "max": 100}]
        errors = check_bounds(df, schema)
        assert len(errors) == 1
        assert "below min" in errors[0].message

    def test_fails_above_max(self):
        df = pd.DataFrame({"price": np.array([10.0, 50.0, 150.0])})
        schema = [{"column_name": "price", "min": 0, "max": 100}]
        errors = check_bounds(df, schema)
        assert len(errors) == 1
        assert "above max" in errors[0].message

    def test_skips_missing_bounds(self):
        df = pd.DataFrame({"x": np.array([1.0, 2.0])})
        schema = [{"column_name": "x"}]  # no min/max
        errors = check_bounds(df, schema)
        assert len(errors) == 0


class TestCheckNulls:
    def test_passes_partial_nulls(self):
        df = pd.DataFrame({"x": pd.array([1.0, None, 3.0])})
        schema = [{"column_name": "x"}]
        errors = check_nulls(df, schema)
        assert len(errors) == 0

    def test_fails_all_nulls(self):
        df = pd.DataFrame({"x": pd.array([None, None, None])})
        schema = [{"column_name": "x"}]
        errors = check_nulls(df, schema)
        assert len(errors) == 1
        assert "entirely null" in errors[0].message

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        schema = [{"column_name": "b"}]
        errors = check_nulls(df, schema)
        assert len(errors) == 0


class TestCheckFormats:
    def test_passes_valid_emails(self):
        df = pd.DataFrame({"email": ["a@b.com", "c@d.org"]})
        schema = [{"column_name": "email", "description": "Email address"}]
        errors = check_formats(df, schema)
        assert len(errors) == 0

    def test_fails_email_missing_at(self):
        df = pd.DataFrame({"email": ["notanemail", "alsono"]})
        schema = [{"column_name": "email", "description": "Email address"}]
        errors = check_formats(df, schema)
        assert len(errors) == 1
        assert "missing '@'" in errors[0].message

    def test_skips_non_email_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        schema = [{"column_name": "name", "description": "Full name"}]
        errors = check_formats(df, schema)
        assert len(errors) == 0


class TestCheckDiversity:
    def test_passes_diverse_values(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        schema = [{"column_name": "name", "data_type": "text"}]
        errors = check_diversity(df, schema)
        assert len(errors) == 0

    def test_fails_all_same(self):
        df = pd.DataFrame({"x": ["same", "same", "same"]})
        schema = [{"column_name": "x", "data_type": "text"}]
        errors = check_diversity(df, schema)
        assert len(errors) == 1
        assert "1 unique value" in errors[0].message

    def test_skips_numeric_columns(self):
        df = pd.DataFrame({"x": [1, 1, 1]})
        schema = [{"column_name": "x", "data_type": "numeric"}]
        errors = check_diversity(df, schema)
        assert len(errors) == 0


class TestValidateIntegration:
    def test_clean_data_passes(self):
        df = pd.DataFrame({
            "year": np.array([2020, 2021, 2022], dtype=np.int64),
            "price": np.array([10.0, 50.0, 100.0]),
            "email": ["a@b.com", "c@d.org", "e@f.com"],
        })
        schema = [
            {"column_name": "year", "data_type": "integer", "min": 2000, "max": 2030},
            {"column_name": "price", "data_type": "numeric", "min": 0, "max": 500},
            {"column_name": "email", "data_type": "text", "description": "Email address"},
        ]
        result = validate(df, schema)
        assert result.passed
        assert len(result.errors) == 0

    def test_bad_data_fails(self):
        df = pd.DataFrame({
            "year": np.array([2020.5, 2021.7, 2022.3]),
            "price": np.array([-10.0, 50.0, 600.0]),
            "email": ["not_an_email", "also_bad", "nope"],
        })
        schema = [
            {"column_name": "year", "data_type": "integer", "min": 2000, "max": 2030},
            {"column_name": "price", "data_type": "numeric", "min": 0, "max": 500},
            {"column_name": "email", "data_type": "text", "description": "Email address"},
        ]
        result = validate(df, schema)
        assert not result.passed
        # Should catch: integer check (3 errors), bounds check (2), format check (3)
        assert len(result.errors) >= 3

    def test_retry_logic_good_first_attempt(self):
        """Validation should pass first time with properly generated data."""
        from datasmith.generation.generator import generate_from_schema
        schema = [
            {"column_name": "year", "data_type": "integer", "min": 2015, "max": 2024},
            {"column_name": "email", "data_type": "text", "description": "Email address"},
        ]
        rng = np.random.default_rng(42)
        df = generate_from_schema(schema, 50, rng)
        result = validate(df, schema)
        assert result.passed, f"Unexpected failures: {[e.message for e in result.errors]}"
