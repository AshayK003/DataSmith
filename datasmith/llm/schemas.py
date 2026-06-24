"""Pydantic models for NL → Schema structured output."""

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """A single column definition extracted from natural language."""
    column_name: str = Field(description="Lowercase snake_case column name")
    data_type: str = Field(
        description="One of: numeric, integer, text, boolean, datetime",
        pattern=r"^(numeric|integer|text|boolean|datetime)$",
    )
    description: str = Field(description="What this column represents")
    distribution_hint: str | None = Field(
        None,
        description=(
            "Distribution hint for numeric columns. "
            "One of: normal, uniform, powerlaw, lognormal, left_skewed. "
            "Null for non-numeric columns."
        ),
    )
    min: float | None = Field(None, description="Minimum plausible value (numeric columns)")
    max: float | None = Field(None, description="Maximum plausible value (numeric columns)")
    mean: float | None = Field(None, description="Average/typical value (numeric columns)")
    null_ratio: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Expected fraction of missing values (0-1)",
    )


class NLDiscoveryResult(BaseModel):
    """Result of NL → Schema discovery."""
    domain: str = Field(description="Target domain (e.g., 'e-commerce', 'healthcare', 'finance')")
    domain_description: str = Field(description="Short description of the domain")
    columns: list[ColumnSchema] = Field(description="Discovered columns", min_length=1)
