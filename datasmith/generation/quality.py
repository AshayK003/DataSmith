"""Quality metrics for synthetic data — KS statistics, null-rate drift, and correlation preservation.

Each metric compares a generated batch against the expected statistical
profile from the schema. All metrics are statistical (no ML, no LLM).

Designed for the iterative batched generation loop: each batch is scored,
and scores feed back into parameter adjustment for the next batch.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ── Reference generation ──────────────────────────────────────────────────

_N_REF = 10_000  # Reference sample size
_REF_SEED = 9999  # Fixed seed for deterministic reference


def _generate_reference(schema: list[dict]) -> dict[str, np.ndarray]:
    """Generate a large reference sample from schema stats.

    Uses a fixed seed so reference is deterministic across runs.
    Only generates numeric columns (the only ones with measurable distributions).
    """
    from datasmith.generation.generator import generate_column

    rng = np.random.default_rng(_REF_SEED)
    ref: dict[str, np.ndarray] = {}
    for col in schema:
        name = col.get("column_name", "")
        dtype = col.get("data_type", "text").lower()
        if dtype in ("numeric", "integer"):
            # Clone col with a stable distribution_hint fallback so the
            # reference doesn't depend on inference heuristics
            col_copy = dict(col)
            if not col_copy.get("distribution_hint"):
                col_copy["distribution_hint"] = "normal"
            try:
                ref[name] = generate_column(name, dtype, col_copy, _N_REF, rng)
            except Exception as exc:
                logger.debug("Could not generate reference for '%s': %s", name, exc)
    return ref


# ── Per-column metrics ────────────────────────────────────────────────────


def _column_ks_stat(series: pd.Series, col_schema: dict,
                    ref: dict[str, np.ndarray]) -> float | None:
    """Two-sample KS statistic comparing batch column vs reference distribution.

    Returns None if the column has no reference or too few non-null values.
    """
    name = col_schema.get("column_name", "")
    if name not in ref:
        return None

    batch_data = series.dropna().values
    if len(batch_data) < 10:
        return None

    ref_data = ref[name]
    if len(ref_data) < 100:
        return None

    stat, _ = scipy_stats.ks_2samp(batch_data, ref_data)
    return round(float(stat), 4)


def _column_null_drift(series: pd.Series, col_schema: dict) -> float:
    """Absolute difference between expected and actual null rate."""
    expected_null = float(col_schema.get("null_ratio", 0.0) or 0.0)
    actual_null = float(series.isna().mean())
    return round(abs(actual_null - expected_null), 4)


# ── Dataset-level metrics ─────────────────────────────────────────────────


def _correlation_diff(df: pd.DataFrame, schema: list[dict],
                      ref: dict[str, np.ndarray]) -> float | None:
    """Mean absolute difference in Spearman correlation between batch and ref.

    Only computed when there are at least 2 numeric columns present in both
    the batch DataFrame and the reference.
    """
    numeric_names = [
        c["column_name"] for c in schema
        if c.get("data_type", "").lower() in ("numeric", "integer")
        and c["column_name"] in df.columns
        and c["column_name"] in ref
    ]
    if len(numeric_names) < 2:
        return None

    batch_corr = df[numeric_names].corr(method="spearman").values
    ref_df = pd.DataFrame({k: ref[k] for k in numeric_names})
    ref_corr = ref_df.corr(method="spearman").values
    return round(float(np.abs(batch_corr - ref_corr).mean()), 4)


# ── Public API ────────────────────────────────────────────────────────────


def compute_batch_quality(df: pd.DataFrame,
                          schema: list[dict]) -> dict:
    """Compute quality metrics for a generated batch.

    Args:
        df: Generated DataFrame (one batch).
        schema: Column schema list (same as passed to generate_from_schema).

    Returns a dict with:
        - ks_<col>: KS statistic for each numeric column
        - null_drift_<col>: null-rate deviation per column
        - corr_diff: mean Spearman correlation difference (if >= 2 numeric cols)
        - ks_mean, ks_max, ks_count: aggregate KS stats
        - null_drift_mean: average null-rate drift across columns
        - quality_score: composite 0–1 score (higher = better)
    """
    ref = _generate_reference(schema)
    metrics: dict = {}

    # Per-column
    for col in schema:
        name = col.get("column_name", "")
        if name not in df.columns:
            continue

        # KS statistic
        if col.get("data_type", "").lower() in ("numeric", "integer"):
            ks = _column_ks_stat(df[name], col, ref)
            if ks is not None:
                metrics[f"ks_{name}"] = ks

        # Null-rate drift
        drift = _column_null_drift(df[name], col)
        metrics[f"null_drift_{name}"] = drift

    # Dataset-level
    corr = _correlation_diff(df, schema, ref)
    if corr is not None:
        metrics["corr_diff"] = corr

    # Aggregates
    ks_vals = [v for k, v in metrics.items() if k.startswith("ks_")]
    metrics["ks_mean"] = round(float(np.mean(ks_vals)), 4) if ks_vals else 0.0
    metrics["ks_max"] = round(float(np.max(ks_vals)), 4) if ks_vals else 0.0
    metrics["ks_count"] = len(ks_vals)

    null_vals = [v for k, v in metrics.items() if k.startswith("null_drift_")]
    metrics["null_drift_mean"] = round(float(np.mean(null_vals)), 4) if null_vals else 0.0

    # Composite score (0–1, higher is better)
    penalties = []
    if metrics["ks_mean"] > 0:
        penalties.append(min(metrics["ks_mean"] * 3.0, 1.0))
    if metrics["null_drift_mean"] > 0:
        penalties.append(min(metrics["null_drift_mean"] * 10.0, 1.0))
    if "corr_diff" in metrics and metrics["corr_diff"] > 0:
        penalties.append(min(metrics["corr_diff"] * 5.0, 1.0))

    metrics["quality_score"] = round(max(0.0, 1.0 - float(np.mean(penalties))), 4) if penalties else 1.0

    return metrics
