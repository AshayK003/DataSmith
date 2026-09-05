"""Data Generator — produce realistic synthetic data from KG schema using numpy/scipy.

Uses distribution hints from the Knowledge Graph
(normal, powerlaw, lognormal, uniform) with column stats (mean, std, min, max).

Zero training time, zero deps beyond numpy/scipy which are already installed.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _coerce_stat(value, default=None):
    """Coerce a stat value (min, max, mean, std, etc.) to float.

    LLMs sometimes return numeric fields as strings (e.g. "0.99" instead of
    0.99). This causes '<' not supported between 'str' and 'float' errors
    in samplers. This helper ensures all stat values are safe floats.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Distribution samplers ─────────────────────────────────────────────────


def _sample_normal(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Normal distribution from KG stats.

    When mean is not provided, infers it from min/max midpoint.
    """
    lo = _coerce_stat(stat.get("min"))
    hi = _coerce_stat(stat.get("max"))
    if lo is not None and hi is not None:
        mid = (lo + hi) / 2.0
    else:
        mid = 0.0
    mean = _coerce_stat(stat.get("mean"), mid)
    std = max(_coerce_stat(stat.get("std"), abs(mid) * 0.1 + 1.0), 0.01)
    data = rng.normal(mean, std, n)
    if lo is not None or hi is not None:
        data = np.clip(
            data,
            lo if lo is not None else -np.inf,
            hi if hi is not None else np.inf,
        )
    return data


def _sample_powerlaw(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Power-law (Pareto) skewed to the right.

    When mean is not provided, infers it from min/max range (~30% of span)
    so columns like price(0.99–500) don't collapse to near-min values.
    """
    lo = _coerce_stat(stat.get("min"), 0.0)
    hi = _coerce_stat(stat.get("max"), 100.0)
    if hi <= lo:
        hi = lo + 100.0
    mean = _coerce_stat(stat.get("mean"))
    if mean is None:
        mean = lo + (hi - lo) * 0.3  # powerlaw peaks near the left tail
    std = max(_coerce_stat(stat.get("std"), abs(hi - lo) * 0.2), 0.01)
    # alpha > 2 makes finite variance
    alpha = max((mean / std) ** 2, 1.5)
    data = rng.pareto(alpha, n) + 1.0
    scale = max(mean / np.mean(data) if np.mean(data) > 0 else 1.0, 0.1)
    data = data * scale + lo
    hi = _coerce_stat(stat.get("max"))
    if hi is not None:
        data = np.clip(data, lo, hi)
    return data


def _sample_lognormal(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Lognormal distribution (always positive, right-skewed)."""
    mean = max(_coerce_stat(stat.get("mean"), 1.0), 0.01)
    std = max(_coerce_stat(stat.get("std"), 0.5), 0.01)
    # Convert moment params to lognormal params
    mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    data = rng.lognormal(mu, sigma, n)
    lo = _coerce_stat(stat.get("min"))
    hi = _coerce_stat(stat.get("max"))
    scale = mean / np.mean(data) if np.mean(data) > 0 else 1.0
    data = data * scale
    if lo is not None or hi is not None:
        data = np.clip(
            data,
            lo if lo is not None else -np.inf,
            hi if hi is not None else np.inf,
        )
    return data


def _sample_uniform(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Uniform distribution between min and max."""
    lo = _coerce_stat(stat.get("min"), 0.0)
    hi = _coerce_stat(stat.get("max"), 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return rng.uniform(lo, hi, n)


def _sample_beta_left_skewed(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Left-skewed (negatively skewed) using beta distribution."""
    mean = _coerce_stat(stat.get("mean"), 0.5)
    lo = _coerce_stat(stat.get("min"), 0.0)
    hi = _coerce_stat(stat.get("max"), 1.0)
    if hi <= lo:
        hi = lo + 1.0
    # Beta distribution with a > b for left skew
    total = max(mean - lo, 1.0)
    ratio = max(mean - lo, 0.01) / max(hi - lo, 0.01)
    a = max(total / (hi - lo) * 5, 1.0)
    b = max(a * (1 - ratio) / max(ratio, 0.01), 1.0)
    data = rng.beta(a, b, n) * (hi - lo) + lo
    return np.clip(data, lo, hi)


_DISTRIBUTIONS = {
    "normal": _sample_normal,
    "powerlaw": _sample_powerlaw,
    "lognormal": _sample_lognormal,
    "uniform": _sample_uniform,
    "left_skewed": _sample_beta_left_skewed,
}


def _generate_numeric_column(col_name: str, data_type: str, stats: dict,
                             n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a numeric column using its distribution hint and stats."""
    dist_hint = (stats.get("distribution_hint") or
                 _infer_distribution(_coerce_stat(stats.get("skewness"), 0)))
    sampler = _DISTRIBUTIONS.get(dist_hint, _sample_normal)

    try:
        data = sampler(n, stats, rng)
    except Exception as e:
        logger.warning(
            "%s: %s sampler failed (%s), falling back to uniform",
            col_name, dist_hint, e,
        )
        data = rng.uniform(
            _coerce_stat(stats.get("min"), 0),
            _coerce_stat(stats.get("max"), 100),
            n,
        )

    # Ensure precision
    precision = _coerce_stat(stats.get("precision"))
    if precision and precision > 0:
        data = np.round(data / precision) * precision

    return data


def _infer_distribution(skewness: float) -> str:
    """Infer a distribution from skewness when no hint is available."""
    if abs(skewness) < 0.3:
        return "normal"
    elif skewness > 1.0:
        return "powerlaw"
    elif skewness > 0.3:
        return "lognormal"
    elif skewness < -0.3:
        return "left_skewed"
    return "normal"


# ── Column type dispatch ──────────────────────────────────────────────────


def generate_column(col_name: str, data_type: str,
                    stats: dict, n: int,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate data for a single column based on its type and KG stats.

    Args:
        col_name: Column name (for logging).
        data_type: One of "text", "numeric", "boolean", "datetime".
        stats: Dict with distribution_hint, mean, std, min, max, skewness.
        n: Number of rows.
        rng: Optional numpy random generator.

    Returns numpy array of length n.
    """
    if rng is None:
        rng = np.random.default_rng()

    dtype = data_type.lower()

    if dtype in ("numeric", "integer"):
        data = _generate_numeric_column(col_name, dtype, stats, n, rng)
        if dtype == "integer":
            data = np.round(data).astype(np.int64)
        return data

    elif dtype in ("text", "string"):
        # Try realistic text profiles first (word banks, ID generators, etc.)
        from datasmith.generation.text_profiles import choose_text_generator
        text_gen = choose_text_generator(col_name, stats.get("description", ""))
        if text_gen:
            return text_gen(n, rng, **stats)
        # Fallback: column-name-templated placeholders
        template = stats.get("template", col_name.replace("_", " ").title())
        return np.array([f"{template} {i+1}" for i in range(n)])

    elif dtype == "boolean":
        ratio = _coerce_stat(stats.get("true_ratio"), 0.5)
        return rng.random(n) < ratio

    elif dtype == "datetime":
        start = stats.get("min_date", "2020-01-01")
        end = stats.get("max_date", "2024-12-31")
        start_ns = np.datetime64(start, "ns")
        end_ns = np.datetime64(end, "ns")
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns
        span_ns = int(end_ns - start_ns)
        offsets = (rng.random(n) * span_ns).astype("timedelta64[ns]")
        return start_ns + offsets

    else:
        logger.warning("Unknown type '%s' for %s, generating text", data_type, col_name)
        return np.array([f"{col_name} {i+1}" for i in range(n)])


def generate_from_schema(columns: list[dict], n: int,
                         rng: Optional[np.random.Generator] = None):
    """Generate a DataFrame from a list of column schema dicts.

    Each column schema dict should have:
    - column_name: str
    - data_type: str (numeric, text, boolean, datetime)
    - distribution_hint: str (optional)
    - mean, std, min, max, skewness, null_ratio (optional stats)

    Args:
        columns: List of column schema dicts.
        n: Number of rows to generate.
        rng: Optional numpy random generator.

    Returns pandas DataFrame.
    """
    import pandas as pd

    if rng is None:
        rng = np.random.default_rng()

    data = {}
    for col in columns:
        name = col.get("column_name", "col")
        dtype = col.get("data_type", "text")
        col_data = generate_column(name, dtype, col, n, rng)
        data[name] = col_data

    df = pd.DataFrame(data)

    # Force text column dtypes to object — pandas auto-promotes
    # string arrays to StringDtype, which breaks downstream numeric
    # comparisons used by the validator and injector
    for col in columns:
        if col.get("data_type") in ("text", "string"):
            name = col.get("column_name", "")
            if name in df.columns:
                df[name] = df[name].astype(object)

    return df
