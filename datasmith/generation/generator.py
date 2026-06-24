"""Data Generator — produce realistic synthetic data from KG schema using numpy/scipy.

Ponytail: no SDV, no PyTorch. Uses distribution hints from the Knowledge Graph
(normal, powerlaw, lognormal, uniform) with column stats (mean, std, min, max).

Zero training time, zero deps beyond numpy/scipy which are already installed.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Distribution samplers ─────────────────────────────────────────────────


def _sample_normal(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Normal distribution from KG stats."""
    mean = stat.get("mean", 0.0)
    std = max(stat.get("std", 1.0), 0.01)
    data = rng.normal(mean, std, n)
    lo = stat.get("min")
    hi = stat.get("max")
    if lo is not None:
        data = np.clip(data, lo, hi)
    return data


def _sample_powerlaw(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Power-law (Pareto) skewed to the right."""
    mean = stat.get("mean", 1.0)
    std = max(stat.get("std", 0.5), 0.01)
    # alpha > 2 makes finite variance
    alpha = max((mean / std) ** 2, 1.5)
    lo = stat.get("min", 0.0)
    data = rng.pareto(alpha, n) + abs(lo) + 1.0
    scale = max(mean / np.mean(data) if np.mean(data) > 0 else 1.0, 0.1)
    data = data * scale + lo
    hi = stat.get("max")
    if hi is not None:
        data = np.clip(data, lo, hi)
    return data


def _sample_lognormal(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Lognormal distribution (always positive, right-skewed)."""
    mean = max(stat.get("mean", 1.0), 0.01)
    std = max(stat.get("std", 0.5), 0.01)
    # Convert moment params to lognormal params
    mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    data = rng.lognormal(mu, sigma, n)
    lo = stat.get("min", 0.0)
    hi = stat.get("max")
    scale = mean / np.mean(data) if np.mean(data) > 0 else 1.0
    data = data * scale
    if lo > 0:
        data = data + lo
    if hi is not None:
        data = np.clip(data, lo, hi)
    return data


def _sample_uniform(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Uniform distribution between min and max."""
    lo = stat.get("min", 0.0)
    hi = stat.get("max", 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return rng.uniform(lo, hi, n)


def _sample_neg_binomial(n: int, stat: dict, rng: np.random.Generator) -> np.ndarray:
    """Left-skewed (negatively skewed) using beta distribution."""
    mean = stat.get("mean", 0.5)
    std = max(stat.get("std", 0.2), 0.01)
    lo = stat.get("min", 0.0)
    hi = stat.get("max", 1.0)
    if hi <= lo:
        hi = lo + 1.0
    # Beta distribution with a > b for left skew
    total = max(mean - lo, 1.0)
    a = max(total / (hi - lo) * 5, 1.0)
    b = max(a * (1 - (mean - lo) / max(hi - lo, 0.01)) / max((mean - lo) / max(hi - lo, 0.01), 0.01), 1.0)
    data = rng.beta(a, b, n) * (hi - lo) + lo
    return np.clip(data, lo, hi)


_DISTRIBUTIONS = {
    "normal": _sample_normal,
    "powerlaw": _sample_powerlaw,
    "lognormal": _sample_lognormal,
    "uniform": _sample_uniform,
    "neg_binomial": _sample_neg_binomial,
}


def _generate_numeric_column(col_name: str, data_type: str, stats: dict,
                             n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a numeric column using its distribution hint and stats."""
    dist_hint = (stats.get("distribution_hint") or
                 _infer_distribution(stats.get("skewness", 0)))
    sampler = _DISTRIBUTIONS.get(dist_hint, _sample_normal)

    try:
        data = sampler(n, stats, rng)
    except Exception as e:
        logger.warning("%s: %s sampler failed (%s), falling back to uniform", col_name, dist_hint, e)
        data = rng.uniform(stats.get("min", 0), stats.get("max", 100), n)

    # Ensure precision
    precision = stats.get("precision")
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
        return "neg_binomial"
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
        # Text columns: generate placeholder values
        template = stats.get("template", col_name.replace("_", " ").title())
        return np.array([f"{template} {i+1}" for i in range(n)])

    elif dtype == "boolean":
        ratio = stats.get("true_ratio", 0.5)
        return rng.random(n) < ratio

    elif dtype == "datetime":
        start = stats.get("min_date", "2020-01-01")
        end = stats.get("max_date", "2024-12-31")
        import pandas as pd
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        span = (end_ts - start_ts).total_seconds()
        offsets = rng.random(n) * span
        return np.array([start_ts + pd.Timedelta(seconds=int(s)) for s in offsets],
                        dtype="datetime64[ns]")

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

    return pd.DataFrame(data)
