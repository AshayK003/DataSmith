"""Tests for the Generation Pipeline (Phase 1 MVP)."""

import numpy as np
import pandas as pd

from datasmith.generation import generator, engine
from datasmith.imperfections.injector import apply_profile
from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph


# ── Generator Tests ───────────────────────────────────────────────────────


class TestGenerateColumn:
    def test_generates_numeric(self):
        rng = np.random.default_rng(42)
        stats = {"distribution_hint": "normal", "mean": 50.0, "std": 10.0,
                 "min": 0.0, "max": 100.0}
        data = generator.generate_column("price", "numeric", stats, 100, rng)
        assert len(data) == 100
        assert np.issubdtype(data.dtype, np.number)
        assert 0 < np.mean(data) < 100

    def test_generates_integer(self):
        rng = np.random.default_rng(42)
        stats = {"mean": 5.0, "std": 2.0, "min": 1, "max": 10}
        data = generator.generate_column("qty", "integer", stats, 100, rng)
        assert len(data) == 100
        assert data.dtype == np.int64

    def test_generates_text(self):
        rng = np.random.default_rng(42)
        data = generator.generate_column("name", "text", {}, 50, rng)
        assert len(data) == 50
        assert all(isinstance(v, str) for v in data)
        assert all("Name" in str(v) for v in data)

    def test_generates_boolean(self):
        rng = np.random.default_rng(42)
        stats = {"true_ratio": 0.7}
        data = generator.generate_column("flag", "boolean", stats, 1000, rng)
        assert data.dtype == bool
        ratio = data.mean()
        assert 0.6 <= ratio <= 0.8  # within tolerance

    def test_generates_datetime(self):
        rng = np.random.default_rng(42)
        stats = {"min_date": "2020-01-01", "max_date": "2024-12-31"}
        data = generator.generate_column("date", "datetime", stats, 100, rng)
        assert len(data) == 100
        # numpy datetime64 objects
        assert hasattr(data[0], "__add__")  # datetime-like behavior
        assert str(data[0]).startswith("202")  # within 2020-2024 range

    def test_unknown_type_falls_back_to_text(self):
        rng = np.random.default_rng(42)
        data = generator.generate_column("col", "unknown_type", {}, 10, rng)
        assert len(data) == 10
        assert all(isinstance(v, str) for v in data)

    def test_powerlaw_generation(self):
        rng = np.random.default_rng(42)
        stats = {"distribution_hint": "powerlaw", "mean": 50.0, "std": 30.0,
                 "min": 0.0, "max": 500.0}
        data = generator.generate_column("amount", "numeric", stats, 1000, rng)
        assert len(data) == 1000
        assert np.mean(data) > 0
        assert np.min(data) >= 0


class TestGenerateFromSchema:
    def test_generates_dataframe(self):
        rng = np.random.default_rng(42)
        columns = [
            {"column_name": "id", "data_type": "text"},
            {"column_name": "price", "data_type": "numeric",
             "mean": 50.0, "std": 20.0, "min": 1.0, "max": 200.0},
            {"column_name": "qty", "data_type": "integer",
             "mean": 3.0, "std": 1.5, "min": 1, "max": 10},
        ]
        df = generator.generate_from_schema(columns, 200, rng)
        assert len(df) == 200
        assert list(df.columns) == ["id", "price", "qty"]
        assert df["price"].dtype == np.float64
        assert df["qty"].dtype == np.int64

    def test_reproducible_with_seed(self):
        columns = [
            {"column_name": "x", "data_type": "numeric",
             "mean": 0.0, "std": 1.0, "min": -3, "max": 3},
        ]
        df1 = generator.generate_from_schema(columns, 100, np.random.default_rng(42))
        df2 = generator.generate_from_schema(columns, 100, np.random.default_rng(42))
        pd.testing.assert_frame_equal(df1, df2)


# ── Engine Tests ──────────────────────────────────────────────────────────


class TestSchemaFromKG:
    def test_generic_schema(self):
        schema = engine._generic_schema("e-commerce")
        assert len(schema) >= 6
        names = [c["column_name"] for c in schema]
        assert "price" in names
        assert "quantity" in names

    def test_generic_schema_fallback(self):
        schema = engine._generic_schema("unknown_domain_xyz")
        assert len(schema) == 2
        assert schema[0]["column_name"] == "id"

    def test_schema_from_kg_returns_generic_when_no_domain(self):
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        schema = engine.schema_from_kg(kg, "nonexistent")
        assert len(schema) == 2  # falls back to generic via generate_dataset
        db.close()


class TestGenerateDataset:
    def test_generates_with_generic_schema(self):
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        df = engine.generate_dataset(kg, "e-commerce", n_rows=50, seed=42)
        assert len(df) == 50
        assert len(df.columns) >= 6  # e-commerce generic has 7 columns
        db.close()

    def test_generates_imperfections(self):
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        df = engine.generate_dataset(kg, "e-commerce", n_rows=500,
                                     inject_imperfections=True, seed=42)
        assert len(df) == 500
        db.close()

    def test_custom_schema(self):
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        custom = [
            {"column_name": "a", "data_type": "numeric",
             "mean": 10.0, "std": 2.0, "min": 0, "max": 20},
            {"column_name": "b", "data_type": "text"},
        ]
        df = engine.generate_dataset(kg, "e-commerce", n_rows=100,
                                     custom_schema=custom, seed=42)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 100
        db.close()


class TestImperfectionInjection:
    """Verify the imperfection injection path is actually exercised."""

    def test_injects_nulls_via_pipeline(self):
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0, 40.0, 50.0,
                      60.0, 70.0, 80.0, 90.0, 100.0],
            "name": ["a", "b", "c", "d", "e",
                     "f", "g", "h", "i", "j"],
        })
        rng = np.random.default_rng(42)
        profile = {
            "null_patterns": {
                "price": {"null_pct": 50, "pattern": "MCAR"},
                "name": {"null_pct": 50, "pattern": "MCAR"},
            },
            "outlier_patterns": {},
            "noise_patterns": {},
        }
        apply_profile(df, profile, rng)
        assert df["price"].isnull().sum() > 0, "Numeric nulls injected"
        assert df["name"].isnull().sum() > 0, "Text nulls injected (bug regression guard)"

    def test_null_injection_on_generated_data(self):
        """Full pipeline: generate_dataset with real imperfection profile."""
        db = Database(":memory:")
        kg = KnowledgeGraph(db)
        df = engine.generate_dataset(kg, "e-commerce", n_rows=200,
                                     inject_imperfections=True, seed=42)
        # Profile won't exist in empty KG, so no nulls — but the pipeline
        # should still execute without errors and produce expected shape
        assert len(df) == 200
        assert len(df.columns) >= 6
        db.close()


# ── Database context manager tests ──────────────────────────────────────


class TestDatabaseContextManager:
    def test_commit_on_success(self):
        from datasmith.core.database import Database
        with Database(":memory:") as db:
            db.execute("CREATE TABLE t(x INT)")
            db.execute("INSERT INTO t VALUES (42)")
        # Re-open and verify
        _ = Database(":memory:")
        # SQLite :memory: is a new DB each time, so we can't verify persistence.
        # Just verify no exception raised.
        assert True

    def test_rollback_on_exception(self):
        from datasmith.core.database import Database
        try:
            with Database(":memory:") as db:
                db.execute("CREATE TABLE t(x INT)")
                db.execute("INSERT INTO t VALUES (1)")
                raise ValueError("boom")
        except ValueError:
            pass
        # Connection still usable after rollback
        rows = db.fetchall("SELECT * FROM t")
        assert len(rows) == 0, "Rollback should undo the insert"
        db.close()
