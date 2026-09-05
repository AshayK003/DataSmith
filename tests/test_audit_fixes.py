"""Regression tests for the Sep 2026 audit fixes (P0/P1/P2).

Each test pins one verified bug so it can never silently return.
"""

import numpy as np
import pandas as pd

from datasmith.generation import generator
from datasmith.generation import pipeline as batched
from datasmith.generation import engine
from datasmith.generation import adjuster
from datasmith.generation import quality as quality_mod
from datasmith.imperfections import profiles
from datasmith.imperfections.injector import inject_outliers
from datasmith.quality.validator import check_formats
from datasmith.llm.client import is_safe_base_url
from datasmith.core.ratelimit import RateLimiter

TINY_SCHEMA = [
    {"column_name": "price", "data_type": "numeric",
     "mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0},
]


def test_seed_none_does_not_crash():
    """engine seed=None previously threw TypeError on validation retry."""
    df = engine.generate_dataset(
        kg=None, domain_name="custom", n_rows=20,
        custom_schema=[dict(c) for c in TINY_SCHEMA],
        inject_imperfections=False, seed=None,
    )
    assert len(df) == 20


def test_pipeline_rejects_bad_counts():
    """total_rows=0 / batch_size=0 previously blew up mid-loop."""
    for kwargs in ({"total_rows": 0}, {"batch_size": 0}):
        try:
            batched.batched_generate(
                kg=None, domain_name="custom",
                custom_schema=[dict(c) for c in TINY_SCHEMA],
                inject_imperfections=False, seed=1, **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"no ValueError for {kwargs}")


def test_pipeline_runs_small_batch():
    """Smoke: batched path works end to end with no KG."""
    df = batched.batched_generate(
        kg=None, domain_name="custom", total_rows=20, batch_size=10,
        custom_schema=[dict(c) for c in TINY_SCHEMA],
        inject_imperfections=False, seed=1,
    )
    assert len(df) == 20


def test_clip_max_only():
    """max-only bounds were silently ignored."""
    rng = np.random.default_rng(7)
    col = {"column_name": "x", "data_type": "numeric", "mean": 1000.0,
           "std": 10.0, "max": 50.0}
    data = generator.generate_column("x", "numeric", col, 500, rng)
    assert (data <= 50.0).all()


def test_powerlaw_mean_near_target():
    """Double-shift pushed the mean far above target."""
    rng = np.random.default_rng(3)
    col = {"column_name": "p", "data_type": "numeric",
           "distribution_hint": "powerlaw",
           "mean": 50.0, "std": 30.0, "min": 0.99, "max": 500.0}
    data = generator.generate_column("p", "numeric", col, 5000, rng)
    assert abs(float(np.mean(data)) - 50.0) < 25.0


def test_lognormal_mean_near_target():
    """The +lo shift inflated the lognormal mean."""
    rng = np.random.default_rng(3)
    col = {"column_name": "lab", "data_type": "numeric",
           "distribution_hint": "lognormal",
           "mean": 100.0, "std": 30.0, "min": 0, "max": 500}
    data = generator.generate_column("lab", "numeric", col, 5000, rng)
    assert abs(float(np.mean(data)) - 100.0) < 30.0


def test_ks_adjustment_is_signed():
    """Overshoot pulls down, undershoot pulls up (was: always down)."""
    schema = [{"column_name": "price", "mean": 50.0}]
    down = adjuster.adjust_schema(
        [dict(schema[0])], {"ks_price": 0.25, "mean_price": 60.0})
    up = adjuster.adjust_schema(
        [dict(schema[0])], {"ks_price": 0.25, "mean_price": 40.0})
    assert down[0]["mean"] < 50.0
    assert up[0]["mean"] > 50.0


def test_merge_missingness_only_analysis():
    """Real analyzer output (missingness key) must merge, not drop."""
    existing = {"null_patterns": {}, "null_correlations": []}
    analysis = {"missingness": {"columns": {
        "age": {"null_pct": 12.0, "pattern": "MAR"}}}}
    merged = profiles.merge_profile(existing, analysis)
    assert merged["null_patterns"]["age"]["null_pct"] == 12.0


def test_search_domains_roundtrip(tmp_path):
    """GET /domains?q= crashed — method did not exist."""
    from datasmith.core.database import Database
    from datasmith.schema.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph(Database(tmp_path / "t.db"))
    kg.upsert_domain("e-commerce", "online retail shop")
    hits = kg.search_domains("retail")
    assert any(dict(h)["name"] == "e-commerce" for h in hits)
    assert kg.search_domains("no-such-domain-xyz") == []


def test_rate_limit_eviction_caps_memory():
    """Unbounded key dict grew forever under key rotation."""
    lim = RateLimiter(max_requests=2, window_seconds=60, max_keys=50)
    for i in range(200):
        lim.check(f"key-{i}")
    assert lim.active_keys <= 50


def test_ssrf_guard():
    """User-supplied LLM base URLs must not reach private targets."""
    assert not is_safe_base_url("http://169.254.169.254/latest")
    assert not is_safe_base_url("http://10.0.0.5:8000/v1")
    assert not is_safe_base_url("file:///etc/passwd")
    assert not is_safe_base_url("http://example.com/v1")  # non-loopback http
    assert is_safe_base_url("https://api.groq.com/openai/v1")
    assert is_safe_base_url("http://127.0.0.1:11434/v1")  # local dev


def test_constant_column_outliers_separate():
    """Degenerate-IQR guard was identity — outliers equalled q3."""
    df = pd.DataFrame({"x": [5.0] * 100})
    profile = {"outlier_patterns": {
        "x": {"direction": "high", "outlier_pct": 10.0}}}
    inject_outliers(df, profile, np.random.default_rng(1))
    assert (df["x"] != 5.0).any()


def test_uuid_ids_pass_validation():
    """Strict PREFIX-NUMBERS regex flagged UUIDs as failures."""
    df = pd.DataFrame({"user_id": [
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    ]})
    schema = [{"column_name": "user_id", "data_type": "text",
               "description": "user id"}]
    errors = check_formats(df, schema)
    assert [e for e in errors if e.column == "user_id"] == []


def test_duplicate_ids_flagged():
    """Without a declared format, IDs still need uniqueness."""
    df = pd.DataFrame({"user_id": ["abc", "abc"]})
    schema = [{"column_name": "user_id", "data_type": "text",
               "description": "user id"}]
    errors = check_formats(df, schema)
    assert any(e.check == "uniqueness_check" for e in errors)


def test_quality_reports_signed_keys():
    """Adjuster inputs mean_<col> / null_signed_<col> must exist."""
    df = pd.DataFrame({"price": [48.0, 52.0, 51.0, 49.0] * 10})
    metrics = quality_mod.compute_batch_quality(df, TINY_SCHEMA)
    assert "mean_price" in metrics
    assert "null_signed_price" in metrics
    assert metrics["null_drift_price"] >= 0  # scoring key stays absolute
