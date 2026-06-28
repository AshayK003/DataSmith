"""Generation Engine — orchestrates schema resolution, generation, and imperfection injection.

The generator produces a DataFrame, the injector modifies it in-place,
and the engine ties them together. No state, no side effects.
For Phase 1 (runs under Streamlit, single user).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from datasmith.imperfections.injector import apply_profile
from datasmith.imperfections.profiles import load_profile_from_kg
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.schema.enricher import enrich_schema
from datasmith.generation.generator import generate_from_schema
from datasmith.quality.validator import validate

logger = logging.getLogger(__name__)

MAX_VALIDATION_RETRIES = 3


def schema_from_kg(kg: KnowledgeGraph, domain_name: str) -> list[dict]:
    """Build a column schema list from the KG for a given domain.

    Returns list of dicts suitable for generate_from_schema().
    Falls back to generic schema if KG has no data for this domain.
    """
    result = kg.get_column_schemas_for_domain(domain_name)
    if result is None:
        logger.warning("Domain '%s' not in KG, returning generic schema", domain_name)
        return _generic_schema(domain_name)
    return result


def _generic_schema(domain_name: str) -> list[dict]:
    """Return a generic column schema for domains with no KG data."""
    defaults = {
        "e-commerce": [
            {"column_name": "order_id", "data_type": "text"},
            {"column_name": "customer_id", "data_type": "text"},
            {"column_name": "product_name", "data_type": "text"},
            {"column_name": "price", "data_type": "numeric",
             "distribution_hint": "powerlaw", "mean": 50.0, "std": 30.0,
             "min": 0.99, "max": 500.0},
            {"column_name": "quantity", "data_type": "integer",
             "distribution_hint": "uniform", "mean": 2.0, "std": 1.5,
             "min": 1, "max": 10},
            {"column_name": "order_date", "data_type": "datetime"},
            {"column_name": "shipping_address", "data_type": "text"},
        ],
        "healthcare": [
            {"column_name": "patient_id", "data_type": "text"},
            {"column_name": "age", "data_type": "numeric",
             "mean": 55.0, "std": 18.0, "min": 0, "max": 100},
            {"column_name": "diagnosis_code", "data_type": "text"},
            {"column_name": "lab_result", "data_type": "numeric",
             "distribution_hint": "lognormal", "mean": 100.0, "std": 30.0,
             "min": 0, "max": 500},
            {"column_name": "admission_date", "data_type": "datetime"},
            {"column_name": "discharge_date", "data_type": "datetime"},
        ],
    }
    return defaults.get(domain_name, [
        {"column_name": "id", "data_type": "text"},
        {"column_name": "value", "data_type": "numeric",
         "mean": 50.0, "std": 20.0, "min": 0, "max": 100},
    ])


get_generic_schema = _generic_schema


def generate_dataset(kg: KnowledgeGraph,
                     domain_name: str,
                     n_rows: int = 100,
                     custom_schema: Optional[list[dict]] = None,
                     inject_imperfections: bool = True,
                     seed: Optional[int] = 42) -> pd.DataFrame:
    """Full generation pipeline: schema → generate → inject → validate → return.

    Args:
        kg: KnowledgeGraph instance.
        domain_name: Target domain (e.g. "e-commerce").
        n_rows: Number of rows to generate.
        custom_schema: Optional custom column schema list. Falls back to KG.
        inject_imperfections: Apply domain imperfection profile after generation.
        seed: Random seed for reproducibility.

    Returns Generated DataFrame.

    Note: after generation the output is validated against the schema.
    If validation fails, the pipeline retries with incremented seeds
    (up to MAX_VALIDATION_RETRIES times).
    """
    try:
        # Step 1: Get schema — None means KG lookup, explicit [] means empty
        schema = schema_from_kg(kg, domain_name) if custom_schema is None else custom_schema
        if not schema:
            schema = _generic_schema(domain_name)
        if not schema:
            raise ValueError(f"No schema found for domain '{domain_name}'")

        # Step 1.5: Enrich schema with semantic constraints
        schema = enrich_schema(schema)

        # Steps 2-4: Generate → Inject → Validate (with retries)
        best_df = None
        last_result = None

        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            current_seed = seed + attempt if attempt > 0 else seed
            rng = np.random.default_rng(current_seed)

            # Step 2: Generate
            df = generate_from_schema(schema, n_rows, rng)

            # Step 3: Inject imperfections
            if inject_imperfections:
                profile = load_profile_from_kg(kg, domain_name)
                if profile:
                    apply_profile(df, profile, rng)

            # Step 4: Validate
            result = validate(df, schema)
            best_df = df
            last_result = result

            if result.passed:
                if attempt > 0:
                    logger.info("Validation passed on retry %d (seed=%d)", attempt, current_seed)
                return df

            logger.warning(
                "Validation failed on attempt %d (seed=%d): %d issue(s)",
                attempt + 1, current_seed, len(result.errors),
            )
            if attempt < MAX_VALIDATION_RETRIES:
                logger.info("Retrying with seed=%d...", current_seed + 1)

        # All retries exhausted — log and return best attempt
        logger.warning(
            "Validation failed after %d attempts. "
            "Returning best effort. Issues:\n%s",
            MAX_VALIDATION_RETRIES + 1, last_result.summary if last_result else "unknown",
        )
        return best_df if best_df is not None else pd.DataFrame()

    except Exception:
        logger.exception("Dataset generation failed")
        raise RuntimeError("An internal error occurred during generation.")
