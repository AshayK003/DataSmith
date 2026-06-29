"""Schema Enricher — adds semantic constraints to raw column schemas.

Takes a basic schema (column_name, data_type, description) produced by
LLM discovery or user input and enriches it with:
- Appropriate data types (integer vs numeric)
- Distribution hints for numeric columns
- Min/max ranges for bounded columns
- Semantic descriptions that help text_profiles match correctly

The enricher only fills missing values — it never overrides an explicit
constraint the user or LLM already set.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Enrichment rules ─────────────────────────────────────────────────────────
# (regex_pattern, enrichment_dict)
# Enrichment values are only applied when the column schema doesn't already
# have that key. More specific patterns come first.

_ENRICHMENT_RULES: list[tuple[re.Pattern, dict[str, Any]]] = [
    # ── Integer columns ──────────────────────────────────────────────────
    # Years — always integer, recent range
    # Matches "year", "year_of_birth", "birth_year", "graduation_year", etc.
    (re.compile(r"(?:^year(?:_|$)|_year$)", re.I),
     {"data_type": "integer", "min": 2015, "max": 2024,
      "distribution_hint": "uniform"}),

    # Ages — integer, reasonable demographic range
    # Matches "age", "customer_age", "age_years", "age_at_admission", etc.
    (re.compile(r"(?:^|_)(?:age|customer_age|patient_age|age_group)(?:$|_)", re.I),
     {"data_type": "integer", "min": 18, "max": 90,
      "distribution_hint": "normal", "mean": 45, "std": 15}),

    # Quantities / counts — integer, right-skewed
    # Matches "quantity", "quantity_sold", "order_qty", "stock_count", "num_items", etc.
    (re.compile(r"(?:^|_)(?:quantity|qty|count|units_sold|items_sold|num)(?:$|_)", re.I),
     {"data_type": "integer", "min": 0, "max": 1000,
      "distribution_hint": "powerlaw"}),

    # ── Numeric columns ──────────────────────────────────────────────────
    # Prices / amounts — right-skewed (powerlaw), broad range
    # Matches "price", "unit_price", "total_amount", "price_per_unit", etc.
    (re.compile(r"(?:^|_)(?:price|cost|fee|fare|revenue|salary|income|"
                r"payment|charge|total|subtotal|balance|budget)(?:$|_)", re.I),
     {"data_type": "numeric", "min": 0.99, "max": 999.99,
      "distribution_hint": "powerlaw"}),

    # Time-based rates (nightly, hourly, daily, etc.) — monetary, not percentages
    (re.compile(r"(?:^|_)(?:nightly|hourly|daily|weekly|monthly|yearly)"
                r"[ _-]?rate(?:$|_)", re.I),
     {"data_type": "numeric", "min": 0.99, "max": 999.99,
      "distribution_hint": "powerlaw"}),

    # Discount rates / percentages — bounded [0, 100]
    (re.compile(r"(?:percent|percentage|pct|discount[_ -]?rate|tax[_ -]?rate|"
                r"interest[_ -]?rate)", re.I),
     {"data_type": "numeric", "min": 0, "max": 100,
      "distribution_hint": "normal", "mean": 25, "std": 15}),

    # Ratings (1-5 or 1-10)
    # Matches "rating", "satisfaction_score", "exam_grade", etc.
    (re.compile(r"(?:^|_)(?:rating|score|grade|rank|stars|satisfaction)(?:$|_)", re.I),
     {"data_type": "numeric", "min": 1, "max": 5,
      "distribution_hint": "normal", "mean": 3.5, "std": 1.0}),

    # ── Text columns — semantic descriptions ────────────────────────────
    # These ensure the text_profiles rules match correctly by overriding
    # the description to include keywords that trigger the right generator.
    (re.compile(r"^(email|e-?mail|email_address|mail)$", re.I),
     {"data_type": "text", "description": "Email address"}),

    (re.compile(r"^(phone|mobile|contact_no|phone_number)$", re.I),
     {"data_type": "text", "description": "Phone number"}),

    (re.compile(r"^url|^website|^domain|^link", re.I),
     {"data_type": "text", "description": "Website URL"}),

    # Name columns — ensure name, not generic text
    (re.compile(r"^(name|full_name|customer_name|user_name|"
                r"first_name|last_name|surname)$", re.I),
     {"data_type": "text", "description": "Customer full name"}),

    # Address — location text
    (re.compile(r"^(address|shipping_address|billing_address|location)$", re.I),
     {"data_type": "text", "description": "Address description"}),

    # Country — ensure description triggers country generator
    (re.compile(r"^country", re.I),
     {"data_type": "text", "description": "Country of residence"}),

    # City
    (re.compile(r"^city", re.I),
     {"data_type": "text", "description": "City name"}),

    # IDs — ensure description triggers ID generators in text_profiles
    (re.compile(r"_?(id|identifier|key)$", re.I),
     {"data_type": "text", "description": "Unique identifier"}),

    # Product/category names
    (re.compile(r"(product|item|merchant|category)", re.I),
     {"data_type": "text", "description": "Product category"}),
]

# Known distribution hints for validation
_VALID_DISTRIBUTIONS = {"uniform", "normal", "powerlaw", "lognormal", "left_skewed"}


def enrich_schema(columns: list[dict]) -> list[dict]:
    """Enrich a list of column schemas with semantic constraints.

    Each column dict may contain: column_name, data_type, description,
    distribution_hint, mean, std, min, max, precision.

    The enricher only fills MISSING values — it never overrides an
    explicit constraint already set in the schema.

    Args:
        columns: Raw column schema list from LLM discovery or user input.

    Returns:
        Enriched column schema list (mutated copies, originals untouched).
    """
    enriched = []
    for col in columns:
        col = {k: v for k, v in col.items() if v is not None}  # strip Nones
        name = col.get("column_name", "")

        # Apply first matching semantic rule
        for pattern, enrichment in _ENRICHMENT_RULES:
            if pattern.search(name):
                for key, value in enrichment.items():
                    if key == "data_type":
                        # Always override data_type — the semantic pattern
                        # knows better than LLM's generic "numeric" output
                        col[key] = value
                    elif key not in col:
                        # Fill missing constraints only
                        col[key] = value
                break  # first match wins

        # ── Post-processing ─────────────────────────────────────────────
        dtype = col.get("data_type", "text").lower()

        # Normalize numeric types
        if dtype in ("float", "double", "real", "decimal", "number"):
            col["data_type"] = "numeric"
        elif dtype in ("int", "int64", "int32"):
            col["data_type"] = "integer"

        # Ensure integer columns have no distribution that produces floats
        # without explicit rounding (uniform integer is fine)
        if col.get("data_type") == "integer":
            dh = col.get("distribution_hint")
            if dh not in ("uniform", None):
                # Only uniform produces clean integers; for others, drop hint
                # so the generator uses normal which gets rounded by integer path
                logger.debug(
                    "Overriding distribution_hint '%s' to 'uniform' for "
                    "integer column '%s'", dh, name,
                )
                col["distribution_hint"] = "uniform"

        # Fill missing min/max for numeric columns when missing
        if col.get("data_type") in ("numeric", "integer"):
            _fill_range(col, name)

        enriched.append(col)

    return enriched


def _fill_range(col: dict, name: str) -> None:
    """Fill missing min/max for numeric columns with sensible defaults."""
    if "min" not in col and "max" not in col:
        dh = col.get("distribution_hint", "normal")
        if dh == "uniform":
            col["min"] = 0
            col["max"] = 100
        elif dh == "powerlaw":
            col["min"] = 0.01
            col["max"] = 1000.0
            if "mean" not in col:
                col["mean"] = 50.0
        elif dh == "lognormal":
            col["min"] = 0.01
            col["max"] = 1000.0
            if "mean" not in col:
                col["mean"] = 100.0
                col["std"] = 50.0
        else:  # normal or left_skewed
            col["min"] = 0
            col["max"] = 100
            if "mean" not in col:
                col["mean"] = 50.0
                col["std"] = 20.0
    elif "min" not in col:
        col["min"] = 0
    elif "max" not in col:
        col["max"] = col["min"] + 100


def _build_semantic_map() -> dict[str, str]:
    """Build a column-name → semantic-type map for quick lookup.
    Useful for debugging and reporting.
    """
    mapping: dict[str, str] = {}
    for pattern, enrichment in _ENRICHMENT_RULES:
        semantic = enrichment.get("data_type", "text")
        # Use the human-readable part of the pattern
        mapping[pattern.pattern] = semantic
    return mapping
