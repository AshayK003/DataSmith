"""Typed DTOs for the data generation pipeline.

Replaces bare dicts as the schema contract between KG, LLM discovery,
and the generator. Lightweight dataclasses — no Pydantic, no new deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ColumnDef:
    """A column definition for the generation pipeline.

    All fields have sensible defaults so partial specs (from KG or LLM)
    produce usable data without explicit validation.
    """

    column_name: str
    data_type: str = "text"

    # Distribution
    distribution_hint: Optional[str] = None  # normal, uniform, powerlaw, lognormal
    mean: float = 0.0
    std: float = 1.0
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: float = 0.0

    # Imperfections
    null_ratio: float = 0.0

    # Numeric precision
    precision: int = 0  # 0 = no rounding

    # Text
    template: str = ""

    # Boolean
    true_ratio: float = 0.5

    # Datetime
    min_date: str = "2020-01-01"
    max_date: str = "2024-12-31"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ColumnDef":
        """Construct from a bare dict (backward compat with engine/KG callers)."""
        return cls(
            column_name=d.get("column_name", "col"),
            data_type=d.get("data_type", "text"),
            distribution_hint=d.get("distribution_hint"),
            mean=d.get("mean", 0.0),
            std=d.get("std", 1.0),
            min=d.get("min"),
            max=d.get("max"),
            skewness=d.get("skewness", 0.0),
            null_ratio=d.get("null_ratio", 0.0),
            precision=d.get("precision", 0),
            template=d.get("template", ""),
            true_ratio=d.get("true_ratio", 0.5),
            min_date=d.get("min_date", "2020-01-01"),
            max_date=d.get("max_date", "2024-12-31"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Export back to dict for backward-compatible tree."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
