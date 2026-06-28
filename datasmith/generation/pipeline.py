"""Batched generation pipeline — the core of Phase 0.

Orchestrates iterative batch generation with quality feedback:
  generate → evaluate → adjust → next batch → concatenate

Each batch is independently generated, quality-checked, and retried if it
falls below the quality threshold. Parameters are adjusted between batches
based on quality metrics, so later batches compensate for drift in earlier ones.

Usage:
    from datasmith.generation.pipeline import batched_generate

    df = batched_generate(kg, "e-commerce", total_rows=5000)
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from datasmith.generation.engine import generate_dataset, schema_from_kg
from datasmith.generation.quality import compute_batch_quality
from datasmith.generation.adjuster import adjust_schema, adjust_imperfection_profile

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_BATCH_SIZE = 1000
_DEFAULT_QUALITY_THRESHOLD = 0.80
_DEFAULT_MAX_RETRIES = 3


def batched_generate(
    kg,
    domain_name: str,
    total_rows: int = 5000,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    custom_schema: Optional[list[dict]] = None,
    inject_imperfections: bool = True,
    quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a dataset using iterative batched generation with quality feedback.

    The pipeline:
        1. Resolves the schema once (from KG, custom, or generic).
        2. Splits total_rows into batches of batch_size.
        3. For each batch:
            a. Generate with current schema + profile.
            b. Compute quality metrics.
            c. If quality < threshold, retry up to max_retries times.
            d. Adjust schema + profile parameters based on quality (feedback).
        4. Concatenate all batches and return.

    Args:
        kg: KnowledgeGraph instance.
        domain_name: Target domain.
        total_rows: Total number of rows to generate.
        batch_size: Rows per batch.
        custom_schema: Optional custom schema (list of column dicts).
        inject_imperfections: Apply domain imperfection profile.
        quality_threshold: Minimum quality_score (0–1) to accept a batch.
        max_retries: Max regeneration attempts per batch.
        seed: Random seed for reproducibility.

    Returns Generated DataFrame with all batches concatenated.
    """
    rng = np.random.default_rng(seed)

    # Step 1: Resolve schema once (consistent across batches)
    schema = (
        custom_schema
        or schema_from_kg(kg, domain_name)
        or _generic_schema(domain_name)
    )
    if not schema:
        raise ValueError(f"No schema found for domain '{domain_name}'")

    # Step 2: Resolve imperfection profile once
    profile = None
    if inject_imperfections:
        from datasmith.imperfections.profiles import load_profile_from_kg
        profile = load_profile_from_kg(kg, domain_name)

    # Step 3: Batch loop
    n_batches = math.ceil(total_rows / batch_size)
    all_batches: list[pd.DataFrame] = []
    total_retries = 0
    quality_log: list[dict] = []

    current_schema = schema
    current_profile = profile

    for i in range(n_batches):
        actual_rows = min(batch_size, total_rows - sum(len(b) for b in all_batches))
        if actual_rows <= 0:
            break

        batch_seed = int(rng.integers(0, 2**31))
        accepted = False

        for attempt in range(max_retries + 1):
            try:
                batch = generate_dataset(
                    kg=kg,
                    domain_name=domain_name,
                    n_rows=actual_rows,
                    custom_schema=current_schema,
                    inject_imperfections=inject_imperfections,
                    seed=batch_seed + attempt,
                )
            except Exception as exc:
                logger.warning("Batch %d attempt %d failed: %s", i, attempt, exc)
                if attempt < max_retries:
                    continue
                raise

            quality = compute_batch_quality(batch, current_schema)
            quality_log.append({
                "batch": i,
                "attempt": attempt,
                **quality,
            })

            if quality["quality_score"] >= quality_threshold:
                accepted = True
                break

            total_retries += 1

        if not accepted and len(all_batches) == 0 and total_retries > 0:
            logger.warning(
                "Batch %d accepted below threshold after %d retries (score=%.3f)",
                i, max_retries, quality["quality_score"],
            )

        all_batches.append(batch)

        # Step 4: Adjust parameters for next batch
        current_schema = adjust_schema(current_schema, quality)
        current_profile = adjust_imperfection_profile(current_profile, quality, current_schema)

    # Step 5: Concatenate
    result = pd.concat(all_batches, ignore_index=True)
    logger.info(
        "Batched generation complete: %d rows in %d batches "
        "(quality=%.3f, retries=%d)",
        len(result), n_batches,
        float(np.mean([q.get("quality_score", 0) for q in quality_log])),
        total_retries,
    )

    return result


def _generic_schema(domain_name: str) -> list[dict]:
    """Fallback generic schema when KG has no data."""
    from datasmith.generation.engine import get_generic_schema
    return get_generic_schema(domain_name)
