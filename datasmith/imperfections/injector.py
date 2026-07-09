"""Imperfection Injector — apply nulls, outliers, noise, and skew to clean DataFrames.

numpy only. No additional deps. Each injector modifies a DataFrame
in-place — the generator produces clean data, the injectors add realism.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def inject_nulls(df, profile: dict, rng: Optional[np.random.Generator] = None) -> None:
    """Inject null values following a domain's missing data profile.

    Supports:
    - MCAR: Uniform random across rows
    - MAR: Conditional on other columns' values
    - MNAR: Conditional on the column's own extreme values

    Args:
        df: DataFrame to modify in-place.
        profile: Domain imperfection profile (with null_patterns).
        rng: Optional numpy random generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    null_patterns = profile.get("null_patterns", {})
    null_corrs = profile.get("null_correlations", [])

    for col, pattern in null_patterns.items():
        if col not in df.columns:
            continue

        # Skip only boolean columns (can't hold NaN).
        # Handle extension dtypes (pandas StringDtype, etc.) safely.
        if pd.api.types.is_bool_dtype(df[col].dtype):
            continue

        null_pct = pattern.get("null_pct", 0) / 100.0
        if null_pct <= 0:
            continue

        # Convert integers to float64 to support NaN
        # Done AFTER null_pct guard to avoid converting when no nulls will be injected
        if pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype(np.float64)

        n = len(df)
        missing_type = pattern.get("pattern", "MCAR")

        if missing_type == "MCAR":
            # Uniform random nulls
            mask = rng.random(n) < null_pct

        elif missing_type == "MAR":
            # Correlated with first other column in the null_correlations
            # Find the first column this one co-nulls with
            related_col = None
            for corr in null_corrs:
                if col in corr.get("cols", []) and len(corr["cols"]) >= 2:
                    related_col = next((c for c in corr["cols"] if c != col), None)
                    break
            if related_col and related_col in df.columns:
                # If related column has values in the extreme, null more often
                try:
                    if pd.api.types.is_numeric_dtype(df[related_col].dtype):
                        series = df[related_col].fillna(df[related_col].median())
                        # Higher values → more likely null
                        spread = max(series.max() - series.min(), 1)
                        normalized = (series - series.min()) / spread
                        # Scale prob: original null_pct * normalized_rank
                        probs = normalized * null_pct * 2
                        probs = np.clip(np.nan_to_num(probs, nan=0.0), 0, 1.0)
                        mask = rng.random(n) < probs
                    else:
                        mask = rng.random(n) < null_pct
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning("Null injection failed (%s), falling back to MCAR", e)
                    mask = rng.random(n) < null_pct
            else:
                mask = rng.random(n) < null_pct

        elif missing_type == "MNAR":
            # Missing correlated with own extreme values
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                try:
                    series = df[col].fillna(df[col].median())
                    z_scores = np.abs((series - series.mean()) / max(series.std(), 1e-6))
                    # Standardize z-scores to [0, 1] and scale by null_pct
                    scaled = z_scores / max(z_scores.max(), 1e-6) * null_pct * 3
                    probs = np.clip(np.nan_to_num(scaled, nan=0.0), 0, 1.0)
                    mask = rng.random(n) < probs
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning("Null injection failed (%s), falling back to MCAR", e)
                    mask = rng.random(n) < null_pct
            else:
                mask = rng.random(n) < null_pct
        else:
            mask = rng.random(n) < null_pct

        df.loc[mask, col] = np.nan


def inject_outliers(df, profile: dict, rng: Optional[np.random.Generator] = None) -> None:
    """Inject outliers following a domain's outlier profile.

    Args:
        df: DataFrame to modify in-place.
        profile: Domain imperfection profile (with outlier_patterns).
        rng: Optional numpy random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    outlier_patterns = profile.get("outlier_patterns", {})

    for col, pattern in outlier_patterns.items():
        if col not in df.columns:
            continue

        outlier_pct = pattern.get("outlier_pct", 0) / 100.0
        if outlier_pct <= 0:
            continue

        if not pd.api.types.is_float_dtype(df[col].dtype):
            if pd.api.types.is_integer_dtype(df[col].dtype):
                df[col] = df[col].astype(np.float64)
            else:
                continue

        n = len(df)
        n_outliers = max(1, int(n * outlier_pct))
        outlier_indices = rng.choice(n, n_outliers, replace=False)

        direction = pattern.get("direction", "both")
        series = df[col].dropna()
        if len(series) < 5:
            continue

        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        base_iqr = max(iqr, abs(q3 - q1) * 0.1)  # Guard against degenerate

        for idx in outlier_indices:
            if direction == "high" or (direction == "both" and rng.random() > 0.5):
                # Outlier above upper bound: q3 + 3-10x IQR
                multiplier = 3 + rng.random() * 7
                df.loc[df.index[idx], col] = q3 + base_iqr * multiplier
            else:
                # Outlier below lower bound: q1 - 3-10x IQR
                multiplier = 3 + rng.random() * 7
                df.loc[df.index[idx], col] = q1 - base_iqr * multiplier


def inject_noise(df, profile: dict, rng: Optional[np.random.Generator] = None) -> None:
    """Inject measurement noise following a domain's noise profile.

    Args:
        df: DataFrame to modify in-place.
        profile: Domain imperfection profile (with noise_patterns).
        rng: Optional numpy random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_patterns = profile.get("noise_patterns", {})

    for col, pattern in noise_patterns.items():
        if col not in df.columns:
            continue

        rounding_pct = pattern.get("rounding_pct", 0)
        precision = pattern.get("precision", 0.01)

        if rounding_pct <= 0:
            continue

        # Convert integers to float64 to support rounding fractions
        if not pd.api.types.is_float_dtype(df[col].dtype):
            if pd.api.types.is_integer_dtype(df[col].dtype):
                df[col] = df[col].astype(np.float64)
            else:
                continue

        # Apply rounding to a subset of non-null values
        series = df[col].dropna()
        if len(series) == 0:
            continue
        n_round = max(1, int(len(series) * rounding_pct / 100))
        round_idx = rng.choice(len(series), n_round, replace=False)
        for idx in round_idx:
            val = series.iloc[idx]
            if not np.isnan(val) and precision > 0:
                df.loc[series.index[idx], col] = round(val / precision) * precision


def apply_profile(df, profile,
                  rng: Optional[np.random.Generator] = None,
                  do_nulls: bool = True,
                  do_outliers: bool = True,
                  do_noise: bool = True) -> None:
    """Apply all relevant imperfections from a domain profile to a DataFrame.

    Modifies the DataFrame in-place. For reproducibility, pass a seeded rng.

    Args:
        df: DataFrame to modify in-place.
        profile: Domain imperfection profile dict.
        rng: Optional numpy random generator.
        do_nulls: Set False to skip null injection.
        do_outliers: Set False to skip outlier injection.
        do_noise: Set False to skip noise injection.
    """
    if do_nulls:
        inject_nulls(df, profile, rng)
    if do_outliers:
        inject_outliers(df, profile, rng)
    if do_noise:
        inject_noise(df, profile, rng)
