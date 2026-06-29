"""LLM Critique Layer — audits generated datasets against the original prompt.

After generation, the LLM reviews the dataset with a professional eye:
1. Identifies columns that don't belong in the original request
2. Flags unrealistic values or wrong data types
3. Suggests fixes (rename, retype, drop, clamp)
4. Returns a cleaned DataFrame matching the user's intent

Only fires when an LLM is available. If not, returns the dataset unchanged.
"""

import json
import logging
import re
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from datasmith.llm.client import chat_complete, is_available

logger = logging.getLogger(__name__)


class CritiqueFix(BaseModel):
    """A single fix instruction from the LLM critique."""

    column: str = Field(description="Column name to fix")
    action: str = Field(
        description="One of: drop, retype, rename, clamp",
        pattern=r"^(drop|retype|rename|clamp)$",
    )
    reason: str = Field(description="Why this fix is needed")
    new_type: Optional[str] = Field(
        None, description="New data_type for retype action"
    )
    new_name: Optional[str] = Field(
        None, description="New column name for rename action"
    )
    clamp_min: Optional[float] = Field(
        None, description="Min clamp value for clamp action"
    )
    clamp_max: Optional[float] = Field(
        None, description="Max clamp value for clamp action"
    )


class CritiqueResult(BaseModel):
    """Structured critique output from the LLM."""

    summary: str = Field(description="One-paragraph professional critique of the dataset")
    issues_found: int = Field(description="Number of issues identified")
    fixes: list[CritiqueFix] = Field(
        default_factory=list,
        description="List of fixes to apply",
    )
    columns_to_drop: list[str] = Field(
        default_factory=list,
        description="Extra columns not in the original prompt that should be removed",
    )


class SchemaVerificationResult(BaseModel):
    """Result of verifying schema completeness against the user prompt."""

    relevant: bool = Field(description="Whether the schema is relevant to the original request")
    issues_found: int = Field(description="Number of issues identified")
    summary: str = Field(description="One-paragraph assessment")
    missing_columns: list[str] = Field(
        default_factory=list,
        description="Columns mentioned in the user request but missing from the schema",
    )
    extra_columns: list[str] = Field(
        default_factory=list,
        description="Columns present in the schema but not relevant to the request",
    )
    type_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested type changes (e.g. 'quantity should be integer not numeric')",
    )


# ── Schema verification prompt ─────────────────────────────────────────────

_VERIFY_PROMPT = (
    "You are a data schema reviewer. Given a user's natural language request "
    "and a proposed column schema, verify the schema is complete and relevant.\n\n"
    "You will receive:\n"
    "1. The original user request (what they wanted)\n"
    "2. The proposed column schema\n\n"
    "Your review criteria:\n"
    "- COMPLETENESS: Does every column the user asked for appear in the schema?\n"
    "- RELEVANCE: Does every schema column belong in this dataset given the request?\n"
    "- TYPE ACCURACY: Are data types appropriate? (quantities = integer, "
    "names = text, prices = numeric)\n\n"
    "Respond with ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "relevant": true,\n'
    '  "issues_found": 0,\n'
    '  "summary": "Brief assessment paragraph",\n'
    '  "missing_columns": [],\n'
    '  "extra_columns": [],\n'
    '  "type_suggestions": []\n'
    "}\n\n"
    "Rules:\n"
    "- Be concise and specific. Only flag real issues, not stylistic preferences.\n"
    "- 'missing_columns' should only include columns the user EXPLICITLY asked for.\n"
    "- 'extra_columns' should only include columns clearly outside the request's scope.\n"
    "- If the schema is complete and relevant, return issues_found=0 and relevant=true.\n"
    "- IMPORTANT: Respond with ONLY valid JSON. No markdown fences, no commentary.\n"
)


def _parse_verification_response(content: str) -> Optional[SchemaVerificationResult]:
    """Parse LLM JSON response into SchemaVerificationResult."""
    text = content.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        candidates = [p.strip() for p in parts if p.strip() and "{" in p]
        if candidates:
            text = max(candidates, key=len)

    try:
        data = json.loads(text)
        return SchemaVerificationResult(**data)
    except (json.JSONDecodeError, Exception):
        pass

    brace_start = text.find("{")
    if brace_start >= 0:
        for brace_end in range(len(text) - 1, brace_start, -1):
            if text[brace_end] == "}":
                candidate = text[brace_start:brace_end + 1]
                try:
                    data = json.loads(candidate)
                    return SchemaVerificationResult(**data)
                except (json.JSONDecodeError, Exception):
                    continue

    logger.warning("Failed to parse schema verification response")
    return None


def verify_schema(
    schema: list[dict],
    user_prompt: str,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Optional[SchemaVerificationResult]:
    """Verify schema completeness and relevance against the user prompt.

    Runs BEFORE generation to catch missing columns or type mismatches
    at the schema level.

    Args:
        schema: Column schema list (after enrichment + user edits).
        user_prompt: Original user description of what they wanted.
        api_key, base_url, model: Optional LLM config overrides.

    Returns:
        SchemaVerificationResult with issues, or None if LLM unavailable.
    """
    if not user_prompt or not is_available():
        return None

    logger.info("Running schema verification for prompt: %.80s", user_prompt)

    # Build schema text
    schema_lines = []
    for col in schema:
        name = col.get("column_name", "?")
        dtype = col.get("data_type", "text")
        desc = col.get("description", "")
        schema_lines.append(f"  - {name} ({dtype}): {desc}")
    schema_text = "\n".join(schema_lines)

    prompt = (
        f"ORIGINAL REQUEST:\n{user_prompt}\n\n"
        f"PROPOSED SCHEMA ({len(schema)} columns):\n{schema_text}\n\n"
        f"Is this schema complete and relevant for the request? Return JSON."
    )

    content = chat_complete(
        system_prompt=_VERIFY_PROMPT,
        user_prompt=prompt,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1000,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    if not content:
        logger.warning("Schema verification LLM call failed")
        return None

    result = _parse_verification_response(content)
    if not result:
        logger.warning("Could not parse schema verification response")
        return None

    logger.info(
        "Schema verification: %s (%d issues, %d missing, %d extra, %d type suggestions)",
        "PASS" if result.relevant else "ISSUES FOUND",
        result.issues_found,
        len(result.missing_columns),
        len(result.extra_columns),
        len(result.type_suggestions),
    )

    return result


_CRITIQUE_PROMPT = (
    "You are a senior data engineer reviewing a synthetic dataset. "
    "Your job is to critique it professionally and identify issues.\n\n"
    "You will receive:\n"
    "1. The original user request (what they wanted)\n"
    "2. The column schema used for generation\n"
    "3. A sample of the generated data (first 10 rows)\n\n"
    "Your review criteria:\n"
    "- COLUMN RELEVANCE: Does every column belong in this dataset given the "
    "original request? Flag any column that was NOT requested and is not "
    "obviously useful context.\n"
    "- DATA TYPE ACCURACY: Are columns typed correctly? (e.g. quantity should "
    "be integer not numeric, status should be text not numeric)\n"
    "- VALUE REALISM: Do values look plausible? (e.g. age > 150 is wrong, "
    "negative prices are wrong, email without @ is wrong)\n"
    "- NAMING QUALITY: Are column names clear and professional?\n"
    "- MISSING COLUMNS: Is anything obviously missing from the request?\n\n"
    "Respond with ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "summary": "Professional critique paragraph",\n'
    '  "issues_found": 0,\n'
    '  "fixes": [\n'
    "    {\n"
    '      "column": "col_name",\n'
    '      "action": "drop|retype|rename|clamp",\n'
    '      "reason": "why",\n'
    '      "new_type": null,\n'
    '      "new_name": null,\n'
    '      "clamp_min": null,\n'
    '      "clamp_max": null\n'
    "    }\n"
    "  ],\n"
    '  "columns_to_drop": ["extra_col"]\n'
    "}\n\n"
    "Rules:\n"
    "- Be concise and specific.\n"
    "- Only flag real issues, not stylistic preferences.\n"
    "- If the dataset is clean, return issues_found=0 with empty fixes.\n"
    "- For 'drop' actions, also add the column to columns_to_drop.\n"
    "- IMPORTANT: Respond with ONLY valid JSON. No markdown fences, no commentary.\n"
)


def _parse_critique_response(content: str) -> Optional[CritiqueResult]:
    """Parse LLM JSON response into CritiqueResult."""
    text = content.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        candidates = [p.strip() for p in parts if p.strip() and "{" in p]
        if candidates:
            text = max(candidates, key=len)

    try:
        data = json.loads(text)
        return CritiqueResult(**data)
    except (json.JSONDecodeError, Exception):
        pass

    brace_start = text.find("{")
    if brace_start >= 0:
        for brace_end in range(len(text) - 1, brace_start, -1):
            if text[brace_end] == "}":
                candidate = text[brace_start:brace_end + 1]
                try:
                    data = json.loads(candidate)
                    return CritiqueResult(**data)
                except (json.JSONDecodeError, Exception):
                    continue

    logger.warning("Failed to parse LLM critique response")
    return None


def _build_sample_prompt(
    user_prompt: str,
    schema: list[dict],
    df: pd.DataFrame,
) -> str:
    """Build the user prompt for the LLM critique."""
    # Schema summary
    schema_lines = []
    for col in schema:
        name = col.get("column_name", "?")
        dtype = col.get("data_type", "text")
        desc = col.get("description", "")
        schema_lines.append(f"  - {name} ({dtype}): {desc}")
    schema_text = "\n".join(schema_lines)

    # Data sample (first 10 rows, all columns)
    sample = df.head(10)
    # Transpose for readability — each column becomes a section
    data_lines = []
    for col_name in sample.columns:
        values = sample[col_name].tolist()
        # Format values concisely
        formatted = []
        for v in values:
            if pd.isna(v):
                formatted.append("null")
            elif isinstance(v, float):
                formatted.append(f"{v:.2f}")
            else:
                formatted.append(str(v)[:50])
        data_lines.append(f"  {col_name}: [{', '.join(formatted)}]")
    data_text = "\n".join(data_lines)

    return (
        f"ORIGINAL REQUEST:\n{user_prompt}\n\n"
        f"SCHEMA ({len(schema)} columns):\n{schema_text}\n\n"
        f"GENERATED DATA SAMPLE (10 rows):\n{data_text}\n\n"
        f"Review this dataset. Identify issues and return JSON."
    )


def critique_dataset(
    user_prompt: str,
    schema: list[dict],
    df: pd.DataFrame,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> tuple[pd.DataFrame, Optional[str]]:
    """Critique a generated dataset and return cleaned version.

    Args:
        user_prompt: Original user description of what they wanted.
        schema: Column schema used for generation.
        df: Generated DataFrame.
        api_key, base_url, model: Optional LLM config overrides.

    Returns:
        (cleaned_df, critique_summary) — cleaned_df has extra columns dropped
        and fixes applied. critique_summary is the LLM's critique text, or
        None if LLM is unavailable.
    """
    if not user_prompt or not is_available():
        return df, None

    logger.info("Running LLM critique on %d columns, %d rows", len(df.columns), len(df))

    prompt = _build_sample_prompt(user_prompt, schema, df)

    content = chat_complete(
        system_prompt=_CRITIQUE_PROMPT,
        user_prompt=prompt,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1500,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    if not content:
        logger.warning("LLM critique call failed, returning dataset unchanged")
        return df, None

    result = _parse_critique_response(content)
    if not result:
        logger.warning("Could not parse critique response, returning dataset unchanged")
        return df, None

    logger.info(
        "Critique complete: %d issues found, %d drops, %d fixes",
        result.issues_found,
        len(result.columns_to_drop),
        len(result.fixes),
    )

    # Apply fixes
    cleaned = df.copy()

    # Step 1: Drop extra columns not in the original prompt
    schema_names = set(c["column_name"] for c in schema)
    all_drops = set(result.columns_to_drop)
    for fix in result.fixes:
        if fix.action == "drop":
            all_drops.add(fix.column)

    for col_name in all_drops:
        if col_name in cleaned.columns:
            # Validate suggested drop exists in schema — ignore hallucinations
            if col_name not in schema_names:
                logger.warning(
                    "LLM suggested dropping unknown column '%s' — ignored", col_name
                )
                continue
            logger.info("Dropping column '%s': not in original request", col_name)
            cleaned = cleaned.drop(columns=[col_name])

    # Step 2: Apply retype fixes
    for fix in result.fixes:
        if fix.action == "retype" and fix.column in cleaned.columns and fix.new_type:
            logger.info("Retyping column '%s' to '%s'", fix.column, fix.new_type)
            if fix.new_type == "integer":
                cleaned[fix.column] = pd.to_numeric(
                    cleaned[fix.column], errors="coerce"
                ).round().astype("Int64")
            elif fix.new_type == "numeric":
                cleaned[fix.column] = pd.to_numeric(
                    cleaned[fix.column], errors="coerce"
                )
            elif fix.new_type == "text":
                cleaned[fix.column] = cleaned[fix.column].astype(str)

    # Step 3: Apply rename fixes
    for fix in result.fixes:
        if fix.action == "rename" and fix.column in cleaned.columns and fix.new_name:
            if fix.new_name not in cleaned.columns:
                logger.info("Renaming column '%s' to '%s'", fix.column, fix.new_name)
                cleaned = cleaned.rename(columns={fix.column: fix.new_name})

    # Step 4: Apply clamp fixes
    for fix in result.fixes:
        if fix.action == "clamp" and fix.column in cleaned.columns:
            if fix.clamp_min is not None or fix.clamp_max is not None:
                logger.info(
                    "Clamping column '%s' to [%.2f, %.2f]",
                    fix.column,
                    fix.clamp_min or float("-inf"),
                    fix.clamp_max or float("inf"),
                )
                numeric = pd.to_numeric(cleaned[fix.column], errors="coerce")
                if fix.clamp_min is not None:
                    numeric = numeric.clip(lower=fix.clamp_min)
                if fix.clamp_max is not None:
                    numeric = numeric.clip(upper=fix.clamp_max)
                cleaned[fix.column] = numeric

    return cleaned, result.summary
