"""Imperfection Analyzer — extract domain-specific data quality fingerprints from real datasets.

Ponytail: numpy + scipy only. No Cleanlab, no PyOD. Stdlib stats where possible.
Each function returns a JSON-serializable dict. No classes — just functions.
"""

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────


def _is_numeric(series) -> bool:
    """Check if a pandas series is numeric (not datetime, not object)."""
    import pandas as pd
    dtype = series.dtype
    return np.issubdtype(dtype, np.number) and not pd.api.types.is_bool_dtype(series)


def _safe_stats(series) -> dict:
    """Compute basic stats for a numeric series, returning None for missing."""
    if not _is_numeric(series) or series.dropna().empty:
        return {}
    s = series.dropna()
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)),
        "min": float(np.min(s)),
        "q25": float(np.percentile(s, 25)),
        "q50": float(np.median(s)),
        "q75": float(np.percentile(s, 75)),
        "max": float(np.max(s)),
        "skew": float(s.skew()) if len(s) > 2 else 0.0,
        "kurtosis": float(s.kurtosis()) if len(s) > 3 else 0.0,
    }


# ── 1. Null Correlation Matrix ────────────────────────────────────────────


def analyze_null_correlations(df, top_k: int = 20) -> list[dict]:
    """Find column pairs that tend to be null together.

    Returns sorted list of {col_i, col_j, co_null_ratio, jaccard}
    where jaccard = |null_i ∩ null_j| / |null_i ∪ null_j|.
    """
    null_mask = df.isnull()
    cols = null_mask.columns.tolist()
    if len(cols) < 2:
        return []

    results = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            ni = null_mask[cols[i]]
            nj = null_mask[cols[j]]
            both = (ni & nj).sum()
            either = (ni | nj).sum()
            if either == 0:
                continue
            ratio_i = both / max(ni.sum(), 1)
            ratio_j = both / max(nj.sum(), 1)
            results.append({
                "col_i": cols[i],
                "col_j": cols[j],
                "co_null_ratio": float(both / len(df)),
                "jaccard": float(both / either),
                "when_i_null_j_null_pct": float(ratio_i * 100),
                "when_j_null_i_null_pct": float(ratio_j * 100),
            })

    results.sort(key=lambda r: r["co_null_ratio"], reverse=True)
    return results[:top_k]


# ── 2. Outlier Distribution ───────────────────────────────────────────────


def analyze_outliers(df, iqr_multiplier: float = 1.5) -> dict[str, dict]:
    """Detect outlier patterns per numeric column using IQR.

    Returns {col_name: {outlier_pct, direction, magnitude_stats, weekend_concentration?, ...}}
    """
    results = {}
    for col in df.columns:
        if not _is_numeric(df[col]):
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0:
            continue

        n_positive = (outliers > q3).sum()
        n_negative = (outliers < q1).sum()

        results[col] = {
            "outlier_pct": float(len(outliers) / len(series) * 100),
            "above_upper_pct": float(n_positive / len(outliers) * 100),
            "below_lower_pct": float(n_negative / len(outliers) * 100),
            "outlier_mean": float(np.mean(outliers)),
            "outlier_std": float(np.std(outliers, ddof=1)) if len(outliers) > 1 else 0.0,
            "bounds": {"lower": float(lower), "upper": float(upper)},
            "direction": "both" if (n_positive > 0 and n_negative > 0)
                         else ("high" if n_positive > 0 else "low"),
        }

        # Weekend concentration check (if there's a datetime column)
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            try:
                dates = pd.to_datetime(df[date_cols[0]], errors="coerce")
                weekend_mask = dates.dt.dayofweek.isin([5, 6])
                outlier_mask = pd.Series(False, index=df.index)
                outlier_mask.loc[series.index] = series.isin(outliers)
                both_mask = weekend_mask & outlier_mask
                wknd_outliers = both_mask.sum()
                total_outliers = outlier_mask.sum()
                if total_outliers > 0:
                    results[col]["weekend_concentration"] = float(wknd_outliers / total_outliers)
            except Exception:
                pass

    return results


# ── 3. Skew Profiles & Distribution Fitting ──────────────────────────────


def analyze_skew(df) -> dict[str, dict]:
    """Fit distributions to numeric columns using scipy.

    Returns {col_name: {distribution, skewness, kurtosis, fitted_params?}}
    """
    from scipy import stats as scipy_stats

    results = {}
    for col in df.columns:
        if not _is_numeric(df[col]):
            continue
        s = df[col].dropna()
        if len(s) < 20:
            continue

        skewness = float(s.skew())
        kurtosis = float(s.kurtosis())

        # Heuristic distribution family based on skewness
        if abs(skewness) < 0.3 and abs(kurtosis) < 0.5:
            dist = "normal"
        elif skewness > 1.0:
            dist = "powerlaw"
        elif skewness > 0.5:
            dist = "lognormal"
        elif skewness < -0.5:
            dist = "neg_binomial"
        else:
            dist = "uniform"

        results[col] = {
            "distribution_hint": dist,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    return results


# ── 4. Missing Data Pattern Classification ────────────────────────────────


def classify_missingness(df, min_correlated: float = 0.3) -> dict[str, Any]:
    """Heuristic MCAR/MAR/MNAR classification per column.

    MCAR: nulls uncorrelated with other columns (no correlation > 0.3)
    MAR: nulls correlated with >= 1 other column
    MNAR: nulls correlated with the column's own values (proxy: extreme values co-occur with nulls)

    Returns {column: {pattern, correlated_with: [...]}, global_summary: str}
    """
    null_mask = df.isnull()
    non_null_mask = ~null_mask
    numeric_cols = [c for c in df.columns if _is_numeric(df[c])]

    results = {}
    for col in df.columns:
        if null_mask[col].sum() == 0:
            continue

        # Check correlation of this column's nulls with other columns' nulls
        corr_with = {}
        for other in df.columns:
            if other == col or null_mask[other].sum() == 0:
                continue
            # Point-biserial correlation: null_of_col vs null_of_other
            try:
                c = null_mask[col].corr(null_mask[other])
                if abs(c) >= min_correlated:
                    corr_with[other] = round(c, 3)
            except Exception:
                pass

        # Check correlation with column's own values (MNAR heuristic)
        # If values are missing when they'd be extreme, that's MNAR
        mnar_hint = False
        if col in numeric_cols:
            try:
                present = df.loc[non_null_mask[col], col]
                missing_idx = null_mask[col][null_mask[col]].index
                if len(present) > 10 and len(missing_idx) > 5:
                    # Compare distribution of present values vs other columns at null points
                    for other in numeric_cols:
                        if other == col:
                            continue
                        other_at_present = df.loc[present.index, other].dropna()
                        other_at_missing = df.loc[missing_idx, other].dropna()
                        if len(other_at_present) > 5 and len(other_at_missing) > 5:
                            from scipy import stats
                            try:
                                _, pval = stats.ks_2samp(other_at_present, other_at_missing)
                                if pval < 0.05:
                                    mnar_hint = True
                                    corr_with[f"{other}_mnar_hint"] = round(pval, 4)
                            except Exception:
                                pass
            except Exception:
                pass

        if len(corr_with) >= 2 and mnar_hint:
            pattern = "MNAR"
        elif len(corr_with) >= 1:
            pattern = "MAR"
        else:
            pattern = "MCAR"

        results[col] = {
            "pattern": pattern,
            "null_pct": float(null_mask[col].mean() * 100),
            "correlated_with": corr_with if corr_with else None,
        }

    # Global summary
    patterns = [v["pattern"] for v in results.values()]
    if patterns:
        from collections import Counter
        counts = Counter(patterns)
        global_summary = counts.most_common(1)[0][0]
    else:
        global_summary = "no_missing"

    return {"columns": results, "global_summary": global_summary}


# ── 5. Noise / Digit Preference ──────────────────────────────────────────


def analyze_noise(df) -> dict[str, dict]:
    """Detect measurement noise and digit preference patterns.

    Looks for:
    - Rounding (excessive .00, .99, .50 endings)
    - Digit preference (Benford's law deviation)
    - Measurement precision (all values are multiples of some unit)

    Returns {col: {rounding_pct, benford_deviation, precision}}
    """
    results = {}
    for col in df.columns:
        if not _is_numeric(df[col]):
            continue
        s = df[col].dropna()
        if len(s) < 20:
            continue

        # Rounding: what % of values end in .00, .99, .50?
        if np.issubdtype(s.dtype, np.floating):
            fracs = s - np.floor(s)
            round_endings = ((np.abs(fracs) < 0.01) |
                             (np.abs(fracs - 0.99) < 0.01) |
                             (np.abs(fracs - 0.50) < 0.01))
            rounding_pct = float(round_endings.mean() * 100)

            # Detect precision (smallest non-zero difference between any two sorted values)
            unique = np.sort(s.unique())
            if len(unique) > 1:
                diffs = np.diff(unique)
                pos_diffs = diffs[diffs > 0]
                precision = float(np.min(pos_diffs)) if len(pos_diffs) > 0 else 0.0
            else:
                precision = 0.0
        else:
            rounding_pct = 0.0
            precision = 1.0

        results[col] = {
            "rounding_pct": rounding_pct,
            "precision": precision,
        }

    return results


# ── Full Analysis ─────────────────────────────────────────────────────────


def analyze_dataset(df, domain_name: str = "",
                    dataset_name: str = "") -> dict:
    """Run full imperfection analysis on a DataFrame.

    Returns a JSON-serializable dict that can be stored as a domain profile.
    """
    result = {
        "domain": domain_name,
        "dataset": dataset_name or "",
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_correlations": analyze_null_correlations(df),
        "outliers": analyze_outliers(df),
        "skew_profiles": analyze_skew(df),
        "missingness": classify_missingness(df),
        "noise": analyze_noise(df),
    }
    return result


def analyze_csv(file_path: str, domain_name: str = "",
                dataset_name: str = "", **kwargs) -> Optional[dict]:
    """Load a CSV and run full imperfection analysis.

    Args:
        file_path: Path to CSV file.
        domain_name: Domain label (e.g. "e-commerce").
        dataset_name: Optional human-readable dataset name.
        **kwargs: Passed to pd.read_csv (e.g. nrows=10000, encoding=...).

    Returns imperfection fingerprint dict, or None on failure.
    """
    import pandas as pd
    try:
        df = pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return None
    return analyze_dataset(df, domain_name=domain_name, dataset_name=dataset_name)


def analyze_kg_datasets(kg, domain_name: str) -> list[dict]:
    """Run imperfection analysis on all CSV datasets for a domain stored in the KG.

    Looks up dataset_schemas by domain, reads the first column's sample_values
    to infer the data, then runs the full analyzer.

    Note: KG stores schemas, not raw data. This function downloads the original
    datasets from their source URLs to do full analysis.

    Returns list of fingerprint dicts (one per dataset).
    """
    fingerprints = []

    # Get all datasets for this domain
    from datasmith.schema.crawler import _download_file, _find_csv_files
    from datasmith.schema.models import DatasetSchema

    domain = kg.get_domain_by_name(domain_name)
    if not domain:
        logger.warning("Domain '%s' not found in KG", domain_name)
        return fingerprints

    datasets = kg.list_datasets(domain_id=domain.id)
    if not datasets:
        logger.warning("No datasets for domain '%s'", domain_name)
        return fingerprints

    for ds in datasets[:5]:  # Limit to 5 datasets per domain for time
        logger.info("Analyzing %s (%s)", ds.dataset_name, ds.source_url)
        try:
            if ds.source == "url":
                import tempfile
                fp = _download_file(ds.source_url,
                                    os.path.join(tempfile.gettempdir(),
                                                 f"datasmith_{ds.dataset_name.replace(' ', '_')}.csv"))
                if fp and fp.endswith(".csv"):
                    result = analyze_csv(fp, domain_name=domain_name,
                                         dataset_name=ds.dataset_name)
                    if result:
                        fingerprints.append(result)
            elif ds.source == "kaggle":
                import kagglehub
                path = kagglehub.dataset_download(
                    "/".join(ds.source_url.split("/")[-2:]))
                csv_files = _find_csv_files(str(path))
                if csv_files:
                    result = analyze_csv(csv_files[0], domain_name=domain_name,
                                         dataset_name=ds.dataset_name)
                    if result:
                        fingerprints.append(result)
        except Exception as e:
            logger.warning("Analysis failed for %s: %s", ds.dataset_name, e)

    return fingerprints
