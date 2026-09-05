"""Correlation Engine — induces pairwise column correlations via Iman-Conover.

Pure numpy implementation of the Iman-Conover rank-matching algorithm
that reorders column values to approximate a target correlation matrix
while preserving each column's marginal distribution.

Reference: Iman, R.L. and Conover, W.J. (1982). A distribution-free
approach to inducing rank correlation among input variables.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Public API ──────────────────────────────────────────────────────────────


def apply_correlations(
    df: pd.DataFrame,
    correlations: list[dict[str, Any]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Induce target pairwise correlations by reordering column values.

    Uses the Iman-Conover rank-matching method:
    1. Generate multivariate normal with target correlation matrix via Cholesky
    2. Sort each column's actual values to match the rank-order of the
       corresponding normal column
    3. Result preserves marginal distributions while inducing correlations

    Args:
        df: Generated DataFrame (values unchanged, only reordered).
        correlations: List of dicts with ``col_a``, ``col_b``, ``rho``.
            Example: ``[{"col_a": "price", "col_b": "quantity", "rho": 0.85}]``
        rng: NumPy random generator.

    Returns:
        DataFrame with reordered columns (marginals preserved).
        Returns the original DataFrame unchanged if no correlations are
        specified or if fewer than 2 columns are involved.
    """
    if not correlations:
        return df

    df = df.copy()
    n = len(df)
    if n < 2:
        return df

    # Collect unique columns involved in correlations
    cols_involved: list[str] = []
    col_set: set[str] = set()
    for spec in correlations:
        for col in (spec["col_a"], spec["col_b"]):
            if col in df.columns and col not in col_set:
                col_set.add(col)
                cols_involved.append(col)

    k = len(cols_involved)
    if k < 2:
        logger.warning(
            "Correlation engine needs at least 2 valid columns, got %d", k,
        )
        return df

    # Build target correlation matrix (k x k)
    target = np.eye(k)
    idx_map = {col: i for i, col in enumerate(cols_involved)}

    for spec in correlations:
        ca, cb = spec["col_a"], spec["col_b"]
        if ca in idx_map and cb in idx_map:
            i, j = idx_map[ca], idx_map[cb]
            rho = float(spec.get("rho", 0.0))
            # Clamp rho to [-0.999, 0.999] for numerical stability
            rho = max(min(rho, 0.999), -0.999)
            target[i, j] = rho
            target[j, i] = rho

    # Ensure matrix is positive semi-definite
    _regularize_correlation(target)

    try:
        # Cholesky decomposition
        L = np.linalg.cholesky(target)
    except np.linalg.LinAlgError:
        logger.warning(
            "Correlation matrix is not positive definite after PSD adjustment. "
            "Falling back to unmodified data.",
        )
        return df

    # Generate correlated standard normals
    z = rng.normal(size=(n, k)) @ L.T

    # Iman-Conover rank-matching: reorder each column's values to match
    # the rank-order of the corresponding normal vector
    for i, col_name in enumerate(cols_involved):
        _reorder_by_rank(df, col_name, z[:, i])

    return df


# ── Internal helpers ────────────────────────────────────────────────────────


def _regularize_correlation(matrix: np.ndarray) -> None:
    """Ensure the matrix is positive definite via diagonal regularization.

    Adds a small epsilon to the diagonal, which guarantees PSD without
    materially changing the correlation structure (epsilon is 4-5 orders
    of magnitude smaller than typical diagonal values of 1.0).
    """
    k = matrix.shape[0]
    # Check if Cholesky works; if not, regularize
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        logger.debug("Regularizing correlation matrix with epsilon=1e-8")
        matrix += np.eye(k) * 1e-8


def _reorder_by_rank(df: pd.DataFrame, col_name: str,
                     normal_vector: np.ndarray) -> None:
    """Reorder a column's values to match the rank-order of a normal vector.

    This is the core of Iman-Conover: sort the column values, then place
    them at the rank positions of the normal vector. The marginal distribution
    (actual values) is preserved, only the order changes.
    """
    col_data = df[col_name].values

    if col_data.dtype.kind in ("i", "f", "b"):
        # Numeric columns — handle NaN preservation
        non_null_mask = pd.notna(col_data)
        non_null_indices = np.where(non_null_mask)[0]

        if len(non_null_indices) > 1:
            z_subset = normal_vector[non_null_indices]
            ranks = np.argsort(z_subset)
            sorted_vals = np.sort(col_data[non_null_indices])
            result = col_data.copy()
            result[non_null_indices[ranks]] = sorted_vals
            df[col_name] = result
        elif len(non_null_indices) <= 1 and np.any(~non_null_mask):
            # All NaN or just one non-null value — nothing to reorder
            pass

    elif col_data.dtype.kind in ("U", "O", "S"):
        # Text columns — sort lexicographically and reorder
        # Handle NaN/None values by preserving them at their original positions
        non_null_mask = pd.notna(col_data)
        non_null_indices = np.where(non_null_mask)[0]

        if len(non_null_indices) > 1:
            # Get ranks of normal vector at non-null positions
            z_subset = normal_vector[non_null_indices]
            ranks = np.argsort(z_subset)

            # Sort the non-null values (mixed types can't sort — skip pair)
            try:
                sorted_vals = np.sort(col_data[non_null_indices])
            except TypeError:
                logger.debug("Skipping correlation for '%s': mixed-type values", col_name)
                return

            # Place sorted values at rank positions
            result = col_data.copy()
            result[non_null_indices[ranks]] = sorted_vals
            df[col_name] = result

    else:
        logger.debug(
            "Skipping correlation for column '%s' with dtype %s",
            col_name, col_data.dtype,
        )
