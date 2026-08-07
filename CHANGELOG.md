# Changelog

## v0.13.0 (2026-08-07)

### Added

- **Critique fix revalidation (revalidate stage)** — LLM critique fixes are now validated by a deterministic, zero-LLM pass before being applied (`_validate_fixes` in `datasmith/llm/critique.py`). Skipped fixes (inverted clamp ranges, coercions that would nullify >30% of a column's values, missing columns) are logged with reasons instead of being applied destructively. Fixes `drop`/`rename` are unchanged — they keep their existing schema-based validation in the apply loop.
- **Full schema-verification diagnostics in logs** — when schema verification finds issues, the log now includes the complete missing/extra/type-suggestion lists instead of a truncated summary, so a failed verification is actionable without re-running.

### Changed

- **LLM critique apply loop uses validated fixes only** — `critique_dataset()` applies the validated subset returned by `_validate_fixes`, so hallucinated or destructive fixes no longer mutate the cleaned dataset.

## v0.12.0 (2026-06-29)

### Added

- **FastAPI REST API** — new `api.py` with programmatic dataset generation endpoints:
  - `POST /generate` — generate a dataset (returns CSV or JSON)
  - `POST /generate/batch` — batched generation for larger datasets
  - `POST /discover` — natural language → schema discovery
  - `GET /domains` — list available domains (with search via `?q=`)
  - `GET /schemas/{domain}` — get enriched schema for a domain
  - `GET /rate-limit` — check current rate limit status
  - `GET /` — health check
  - Auto-documented OpenAPI at `/docs`
  - Rate-limited via the same in-memory sliding window as the UI
  - Configurable via `DATASMITH_RATE_MAX` and `DATASMITH_RATE_WINDOW` env vars
  - Start with `uv run uvicorn api:app --host 0.0.0.0 --port 8000`
- **Rate limiting** — sliding window rate limiter (`datasmith/core/ratelimit.py`) applied to both the Streamlit UI and REST API:
  - Thread-safe via `threading.RLock`
  - Configurable max requests per time window
  - Friendly error messages when rate limited
  - Default: 10 requests/minute per session/IP
  - Per-key tracking with automatic window expiry
- **Rate limiter tests** — 10 tests covering bounds, concurrency, reset, key isolation, window expiry, and active key tracking.

### Changed

- **Streamlit UI generation now rate-limited** — per-session rate limit check before every generation. Shows a clear error message when the limit is reached.

## v0.11.0 (2026-06-29)

### Fixed

- **`StringDtype` crash (`'<' not supported between instances of 'str' and 'float'`)** — pandas auto-promotes string arrays to `StringDtype`, which causes `TypeError` when `check_bounds` in the validator compares them with numeric min/max bounds. `generate_from_schema()` now explicitly casts all text columns to `object` dtype after DataFrame creation, preventing the leak. This was the root cause of generation failures with text-heavy schemas.
- **Integer columns skipped range validation in UI** — the "Generate Dataset" button only validated `min < max` for `numeric` columns, not `integer`. Integer columns with invalid ranges (e.g. `min` >= `max`) would generate silently bad data. Validation now covers both types.
- **Generator fallback path bypassed `_coerce_stat`** — when a distribution sampler fails and generation falls back to uniform, the `stats.get("min", 0)` call read raw schema values without type coercion. If the LLM returned string-typed stats, this crashed with `rng.uniform("0", ...)`. Now wraps min/max in `_coerce_stat()`.
- **`amount` removed from price enricher rule** — the regex pattern `amount` in the price rule matched compound names like `coverage_amount`, `loan_amount`, `policy_amount`, incorrectly assigning retail price ranges (0.99–999.99) to columns that are typically integer counts or high-value floats. `amount` removed from the price rule — these columns now fall through to generic numeric enrichment.
- **Exception chain broken in `generate_dataset`** — the outer `except Exception` handler swallowed the original exception and raised a generic `RuntimeError` without chaining it, making every generation failure show "An internal error occurred during generation." with zero diagnostic information. Now uses `raise ... from e` to preserve the chain.
- **Format validator false positives with `pd.NA`** — `check_formats` called `.astype(str)` on `StringDtype` columns, converting `pd.NA` values to string `"<NA>"` which then matched format checks incorrectly. Null-representation strings are now filtered out before validation.
- **Duplicate `verify_schema()` call removed from engine** — `generate_dataset()` had the same `verify_schema()` LLM call that already runs in `batched_generate()`. Removed the duplicate to halve token spend per generation.

### Changed

- **LLM critique drop suggestions are now validated** — `critique_dataset()` verifies that columns named in `columns_to_drop` actually exist in the schema before dropping. Hallucinated column names from the LLM are logged and skipped instead of silently modifying the dataset.

## v0.10.0 (2026-06-29)

### Added

- **Expanded text generation coverage** — `text_profiles.py` now covers more column types:
  - Medical: `diagnosis` (20 conditions), `blood_type` (A+, O-, AB+, etc.)
  - Education: `major` (20 fields of study), `student_id` (STU-XXXXXX)
  - Hospitality: `hotel_name` (12 brand names), `booking_id` (BKG-XXXXXX)
  - Support: `ticket_id` (TCK-XXXXXX), `priority` (Critical/High/Medium/Low)
  - Staff: `assigned_to`, `reported_by` now generate real names instead of sentences
- **Time-based rate support in schema enricher** — `nightly_rate`, `hourly_rate`, `daily_rate`, `weekly_rate`, `monthly_rate`, and `yearly_rate` columns are correctly identified as price-like (powerlaw, max=999.99) instead of being treated as percentages (min=0, max=100)

### Fixed

- **Name columns no longer generate sentences** — `full_name`, `first_name`, `last_name`, `customer_name`, and `user_name` now correctly produce first+last names. Root cause: `choose_text_generator()` was replacing underscores with spaces, breaking anchored regex patterns that use underscores.
- **Email columns no longer return city names** — `email_address` columns now produce proper email addresses (e.g. `user@domain.com`). Root cause: the address rule fired before the email rule in `_TEXT_RULES`. Email rule moved ahead of address rule.
- **Blood type hijacked by generic type rule** — `blood_type` now shows real blood types instead of single-letter category labels. Root cause: the generic `(categ|type|class|kind|segment)` rule matched "type" in "blood_type" first. Blood type rule moved before the generic rule.
- **Integer dtype preserved with zero-rate imperfections** — `inject_nulls`, `inject_outliers`, and `inject_noise` no longer convert integer columns to `float64` when their respective injection rates are 0%. Conversion now happens only when nulls/outliers/noise are actually placed.
- **Enricher rate filter narrowed** — the bare `rate` keyword was removed from the percentage detection pattern, preventing `nightly_rate`, `hourly_rate`, etc. from being capped at [0, 100]. Time-based rates are now routed to the price rule instead.

## v0.9.0 (2026-06-28)

### Added

- **LLM Critique Layer** — new module `datasmith/llm/critique.py` that audits generated datasets against the original user prompt before displaying results. When an LLM is available and the user described their dataset in natural language, the critique:
  - Reviews every column for relevance to the original request
  - Drops extra columns not in the prompt that aren't useful context
  - Fixes data type mismatches (e.g. retying "quantity" from numeric to integer)
  - Clamps unrealistic values (e.g. age > 150, negative prices)
  - Renames unclear column names to be more professional
  - Returns a structured critique summary alongside the cleaned DataFrame
  - Falls back gracefully — if LLM is unavailable, the dataset is returned unchanged
- **`user_prompt` parameter** on `generate_dataset()` and `batched_generate()` — when provided, triggers the critique layer. The frontend passes the NL description automatically.

### Changed

- Generation pipeline is now: `schema → enrich → generate → correlate → inject → **critique** → validate → return`
- In batched mode, critique runs **once** on the final concatenated result (not per batch) for efficiency.

## v0.8.0 (2026-06-28)

### Fixed

- **`'<' not supported between instances of 'str' and 'float'`** — LLMs sometimes return numeric schema fields (`min`, `max`, `mean`, `std`) as strings (e.g. `"0.99"` instead of `0.99`). Added `_coerce_stat()` helper in `generator.py` that converts all stat values to `float` with a safe fallback. Applied across all 5 distribution samplers (normal, powerlaw, lognormal, uniform, beta) and `_generate_numeric_column`. Also applied to `skewness`, `precision`, and `true_ratio`.
- **AG Grid string values** — the schema editor's grid cells could return strings for numeric columns after user edits. Added `_safe_float()` in `pages/01_Generate.py` to coerce grid values before they reach the generator.

### Changed

- **LLM datatype decision step** — the NL → Schema system prompt now instructs the LLM to explicitly decide each column's data type *before* generating stats, with a clear decision process: whole numbers → `integer`, decimals → `numeric`, text → `text`. Added `type_reasoning` field to `ColumnSchema` Pydantic model so the LLM explains why it chose each type (e.g. "quantity is always a whole number" or "price can have cents"). This reduces LLMs defaulting everything to `"numeric"` when `"integer"` would be more correct.

## v0.7.0 (2026-06-28)

### Added

- **Correlation Engine** — new module `datasmith/generation/correlator.py` using the Iman-Conover rank-matching method.
  - Induces pairwise column correlations by reordering values, preserving each column's marginal distribution.
  - Accepts arbitrary correlation specs: `[{"col_a": "price", "col_b": "quantity", "rho": 0.85}]`
  - Handles numeric, text, and datetime columns with NaN preservation.
  - PSD regularization via diagonal epsilon for numerically stable Cholesky decomposition.
- **More varied text generation** — expanded and improved `text_profiles.py`:
  - Name pools doubled (20→60 first names, 20→66 last names) for 3,960 unique name combinations
  - City pool expanded (30→105) covering 24 countries
  - New word banks: 25 companies, 25 job titles, 15 departments, 15 products
  - Sentence generator for unknown text columns — 180 pre-composed pattern variations producing realistic-looking descriptions instead of "Placeholder 1"
  - Description/notes columns now produce varied sentences instead of sequential templates
  - Status pool expanded (6→12) with states like "Approved", "Processing", "Delivered"
  - Payment methods pool expanded (10→15) with Google Pay, Apple Pay, PayPal, etc.
  - Catch-all `.*` rule ensures every text column gets a realistic generator
- **9 new correlator tests** covering correlation induction, marginal preservation, multiple pairs, NaN handling, text columns, and full-pipeline integration.

### Changed

- `generate_dataset()` now accepts an optional `correlations` parameter applied after generation and before imperfection injection.
- `engine.py` pipeline: `schema → enrich → generate → **correlate** → inject → validate → return`.

## v0.6.0 (2026-06-28)

### Added

- **Quality Validator** — new module `datasmith/quality/validator.py` that validates generated DataFrames against column schema. Runs 5 quality checks on every generation:
  - **Integer check** — columns typed as `integer` must have no fractional values
  - **Bounds check** — values stay within declared `min`/`max` ranges
  - **Null check** — flags columns that are entirely null
  - **Format check** — email columns must contain `@`, phone columns must have 5+ digits
  - **Diversity check** — text columns must not all be identical (catches generation failures)
- **Auto-retry in `generate_dataset()`** — after generation, the pipeline validates the output and retries with an incremented seed (up to 3 attempts) if quality gates fail. The best-effort result is returned even if all retries are exhausted.
- **`ValidationResult` and `ValidationError`** — public data classes for inspecting validation results programmatically.
- **22 new tests** covering all validator checks, edge cases (extension dtypes, nulls, missing columns), and integration with the generation pipeline.

## v0.5.0 (2026-06-28)

### Added

- **Semantic Schema Enricher** — new module `datasmith/schema/enricher.py` that enriches raw column schemas with semantic constraints. Integrated into `generate_dataset()` between schema resolution and generation. Automatically:
  - Sets `data_type="integer"` for year, age, quantity, and count columns (was "numeric", producing decimals)
  - Sets `data_type="numeric"` with `distribution_hint="powerlaw"` for price/amount columns
  - Sets distribution hints, min/max ranges, and mean/std for known column types
  - Adds semantic descriptions that help text_profiles match email, phone, name, and ID columns

### Changed

- **`enrich_schema(columns)`** — public API for enriching column schemas. Takes a list of column dicts, returns enriched copies. Only fills missing constraints; always overrides `data_type` when a semantic pattern matches (since pattern-based inference is more accurate than LLM's generic "numeric" output).

### Fixed

- **Years, ages, quantities no longer have decimal values** — the enricher assigns `data_type="integer"` to semantic integer columns, and the generator now produces `int64` arrays for these types instead of `float64`.

## v0.4.2 (2026-06-28)

### Fixed

- **Email columns now generate proper emails** — fixed rule ordering in `_TEXT_RULES` where `address` rule (matching "Email address" description) fired before `email` rule. Also changed `choose_text_generator` to match column name first, then description only as fallback, preventing description keywords from hijacking columns.
- **Formatted IDs for customer/user/employee columns** — ID patterns now use `[ _-]?` separator to handle column names normalized to spaces (e.g. `customer_id` → `customer id`). Previously only matched underscore, causing `customer_id` to fall through to template fallback ("Customer Id 1").
- **Powerlaw sampler no longer collapses to near-min** — when `mean` is not provided in the schema, infers it from min/max range (~30% of span). Previously defaulted to `mean=1.0`, producing values near min for any range (e.g. `price(0.99–500)` → ~2).
- **Normal sampler infers mean from min/max** — when `mean` is not specified but `min` and `max` are, defaults to their midpoint instead of 0.0, preventing all values from clipping to the minimum.
- **Expanded country word bank** — from 8 entries to 195+ countries, providing realistic diversity for country columns.
- **New ID patterns** — added `(policy|claim)_(id|num|number)` → `POL-XXXXXXX`.

## v0.4.1 (2026-06-28)

### Fixed

- **pandas StringDtype compatibility** — `np.issubdtype()` crashes on pandas extension dtypes (StringDtype, etc.) in numpy 2.x. Replaced all dtype checks in the imperfection injector with `pd.api.types` equivalents. The generator also ensures text columns use `object` dtype. Fixes 22 failing tests.
- **Lazy kagglehub import** — moved `import kagglehub` from module-level to inside `_crawl_kaggle()`, so test collection and cold-start imports no longer require kagglehub to be installed.
- **Word-boundary domain matching** — `discovery.py` now uses `\b` word boundaries instead of substring matching, preventing false positives like "finance" matching "financing data" or "energy" matching "energy drinks".
- **Empty custom_schema now respected** — `generate_dataset(custom_schema=[])` no longer falls through to KG lookup. Pass `None` for KG fallback, `[]` for no columns.
- **LLM cache corruption guard** — `llm_cache_get()` now wraps `json.loads()` in try/except, returning None on corruption instead of crashing the discovery pipeline.
- **Extension dtype safety in analyzer** — `_is_numeric()` wraps `np.issubdtype` in TypeError guard to handle StringDtype and other extension dtypes.
- **Ensured frictionless installed** — was missing in the active venv, blocking test collection.

## v0.4.0 (2026-06-28)

### Added

- **LLM provider presets** — dropdown in the Generate page's LLM Configuration expander that auto-fills the Base URL and Model for Gemini, Groq, OpenRouter, and OpenCode Zen. Just select your provider and paste your API key. Custom option for manual entry.
- **Custom API key input in frontend** — password field, base URL, and model fields in an `st.expander` on the Generate page. Users can enter any OpenAI-compatible API key at runtime without setting environment variables. Overrides env-var config for the session.
- **Provider retry fallback** — `chat_complete()` now retries the API call without `response_format` if the first attempt fails. Handles providers (like Gemini and OpenCode Zen) that don't support JSON-mode structured output.
- **Robust JSON parsing** — `_parse_llm_response()` handles markdown fences anywhere in the response, extra commentary around JSON, and falls back to searching for any `{...}` block that validates against the Pydantic schema.

### Fixed

- **Gemini quota errors now logged** — the actual API response body (status code + message) is included in the log output when an LLM request fails, making it easy to diagnose quota exhaustion, invalid keys, or unsupported parameters.
- **`null` keys stripped from retry body** — `{"response_format": null}` was sent in the retry body (could confuse picky providers). Now `None`-valued keys are removed entirely before serialization.

### Added

- **Batched iterative generation (Phase 0)** — new `generation/pipeline.py` orchestrator generates data in batches with per-batch quality feedback. KS statistics, null-rate drift, and correlation preservation are measured on each batch. Low-quality batches are retried automatically. Parameters adjust between batches to compensate for sampling drift.
- **Quality metrics module** (`generation/quality.py`) — `compute_batch_quality()` returns per-column KS stats, null-rate drift, and a composite 0–1 quality score. Uses two-sample KS test vs a deterministic reference distribution. No ML, no LLM, no new dependencies.
- **Parameter adjuster** (`generation/adjuster.py`) — proportional-correction rules that tweak column means and null rates between batches based on quality feedback. Original schema is never mutated.
- **UI toggle** on the Generate page — "Iterative quality enhancement" checkbox (on by default) switches between the new batched pipeline and the original single-pass generation.
- **20 new tests** covering quality metrics, adjuster logic, and batched integration — all pass alongside 94 existing tests with zero regressions.

## v0.3.8 (2026-06-24)

### Fixed

- **GitHub Actions CI:** `seed-crawl` job now has `permissions: contents: write` for git push. Git config (user.name/user.email) set in a step *before* the crawl script runs to prevent "fatal: empty ident name" on scheduled runs.
- **Linting:** Fixed all 42 flake8 errors — removed 6 unused imports, 3 dead variables, fixed 1 undefined name (`pd`), wrapped 20 long lines with `# noqa: E501` or string concatenation, cleaned trailing whitespace and EOF newlines.
- **`analyzer.py`:** Added missing `import pandas as pd` (was silently crashing on F821).

### Changed

- **`pyproject.toml`:** Added `[tool.flake8]` section (max-line-length=100, per-file-ignores for data URL lines).
- **`_SYSTEM_PROMPT` in `discovery.py`:** Converted from triple-quoted string to concatenated strings for lint compliance without changing prompt text.

## v0.3.8 (2026-06-24)

### Added

- **Parquet export** — new download button alongside CSV and JSON. pyarrow already ships as a transitive Streamlit dependency, so zero new install weight.

### Fixed

- **Version string drift** — `app.py` footer and `02_About.py` both updated from v0.3.3 → v0.3.7 to match `pyproject.toml`.

## v0.3.7 (2026-06-24)

### Removed

- **`_crawl_huggingface` function** (43 lines) — dead code, no SEED_DATASETS entry uses HuggingFace as source. Falls through to "skipped" cleanly.
- **7 unused Lucide SVG icons** — `HOME`, `SEARCH`, `FOLDER`, `REFRESH`, `SLIDERS`, `CHECK`, `BAR_CHART` removed from `icons.py`. Not imported or referenced anywhere.
- **`docs/PERFORMANCE.md`** (647 lines) — stale AI-generated performance plan, never referenced by code or docs.
- **17 unused imports** across 10 source/test files — `json`, `pathlib.Path`, `typing.Any`, `typing.Optional`, `pytest`, `pandas`, `numpy.noqa`.

### Cleanup

- Net **−750 lines** across 12 files. All 94 tests pass unchanged.

## v0.3.6 (2026-06-24)

### Fixed (3 bugs found via stress-test harness)

- **MAR null correlation crash** — `inject_nulls` raised `IndexError` when `null_correlations` contained a self-referencing entry (e.g., `cols=["x", "x"]`). Used `next(gen, None)` instead of `list[0]` for the related column lookup (`datasmith/imperfections/injector.py:62`)
- **All-NaN noise crash** — `inject_noise` raised `ValueError` on `rng.choice(0, 1)` when a column had zero non-null values. Added empty-Series guard (`datasmith/imperfections/injector.py:180`)
- **Reversed datetime range** — `generate_column` produced dates outside the intended range when `max_date < min_date` (negative span_ns). Added bounds swap guard (`datasmith/generation/generator.py:174`)
- **Noise rounding_pct=0** — was silently rounding 1 value due to `max(1, int(...))`. Now rounds 0 values when pct=0.

### Tests

- **94 tests (was 90)** — 4 new regression tests added to `test_injector_edge.py` and `test_generation_edge.py`

## v0.3.5 (2026-06-24)

### Security (7 fixes applied from external audit)

- **LLM input sanitization** — control characters stripped, XML isolation tags around user input, anti-injection system prompt instruction added (`datasmith/llm/discovery.py`)
- **Removed `allow_unsafe_jscode=True`** — AG Grid no longer permits JavaScript execution in cell renderers, closing a stored XSS vector (`pages/01_Generate.py`)
- **Column name sanitization** — non-alphanumeric/non-whitespace characters stripped, length capped at 128 chars, blank names default to "column" (`pages/01_Generate.py`)
- **CSV formula injection prevention** — cells starting with `=`, `+`, `-`, or `@` are prefixed with `'` before export to prevent Excel formula execution (`pages/01_Generate.py`)
- **LLM rate limiting** — 5-second session-level cooldown between discovery calls to prevent API credit burn (`pages/01_Generate.py`)
- **Safe error messages** — generic user-facing error on generation failure; full exception logged with stack trace (`datasmith/generation/engine.py`)
- **Parameterized PRAGMA query** — `PRAGMA user_version` now uses parameterized `?` placeholder instead of f-string (`datasmith/schema/knowledge_graph.py`)

### Changed

- **Schema editor replaced** — `st.data_editor` (lost edits on re-run) and per-column form widgets both replaced with **AG Grid** (`streamlit-aggrid`). Editable cells, dropdown type editor, built-in sorting/filtering, stable state across re-runs. Delete Selected reads persisted grid component value directly from `st.session_state` for reliable row removal.
- **Delete button styled red** — CSS override for danger styling on the Delete Selected button.
- **90 tests** (unchanged) — all pass.
- **Dependency added:** `streamlit-aggrid>=1.2.1.post2`

## v0.3.4 (2026-06-24)

### Changed
- **Version strings synced** — `pyproject.toml` → 0.3.3, `app.py` + `About.py` → v0.3.3. All three are now consistent.
- **`_sample_neg_binomial` renamed** → `_sample_beta_left_skewed` (uses beta distribution, not negative binomial). Dict key updated to `"left_skewed"`.
- **`_generic_schema` made public** → added `get_generic_schema` alias. Page import updated.

### Removed
- **3 dead imports** — `import json` and `from crawler import SEED_DOMAINS` from `engine.py`; `import time` from `knowledge_graph.py`.

### Fixed
- **Stale docstring** in `get_column_schemas_for_domain()` — removed reference to the old "maximum→max" mapping that was fixed in v0.3.3.

## v0.3.3 (2026-06-24)

### Fixed
- **`generator.py: if lo > 0`** — lognormal min-offset branch only activated when `lo > 0`, silently ignoring negative min values. Changed to `if lo is not None`.
- **`profiles.py: falsy-zero skip`** — `if current and incoming` skipped null_pct averaging when either value was exactly 0. Changed to `is not None` check.
- **`injector.py: NaN probability masks`** — constant columns produced NaN probs in MAR/MNAR paths, causing `rng.random(n) < NaN` to silently skip injection. Added `np.nan_to_num(..., nan=0.0)`.
- **`analyzer.py: reindex_like misalignment`** — weekend-concentration check used `dropna`-subset index for outlier mask while aligning to full DataFrame index. Replaced with properly initialized `pd.Series(False, index=df.index)`.
- **`knowledge_graph.py: dead maximum→max mapping`** — key tuple used `"maximum"` but DB column is `max`. The explicit mapping block was dead code, silently dropping `max` values from the KG into the generator. Changed key to `"max"` in the generic loop.
- **`injector.py: integer columns skipped`** — `inject_outliers` and `inject_noise` rejected integer columns entirely. Now convert to float64 (same pattern as `inject_nulls`).

### Changed
- **Sidebar removed** — version caption (`DataSmith v0.3.1`) moved from sidebar to main dashboard footer, next to the support badge.
- **`components.py` header nav** — now rendered on all pages (Home, Generate, About) via shared `render_header()`.

## v0.3.2 (2026-06-25)

### Added
- **`.streamlit/config.toml`** -- theme tokens, hides Streamlit chrome (hamburger, toolbar, header).
- **Shared icon module** (`datasmith/ui/icons.py`) -- all Lucide SVGs in one place. No more duplicated SVGs across pages.
- **CSS injection** -- design tokens (color, radius, spacing), component overrides (buttons, inputs, tabs, expanders, metrics), responsive rules, keyboard focus ring. Single `@st.cache_resource` block.
- **Form validation** -- min/max range check and negative std check before generation. Shows actionable error per column.
- **Keyboard shortcut** -- Ctrl+Enter triggers the primary Generate button.
- **Empty state guidance** -- domain info shows dataset count or "No datasets crawled yet" instead of blank.

### Fixed
- **Version strings** -- sidebar and About page updated from v0.2.0 to v0.3.1.
- **Accessibility** -- text input now has visible label + help tooltip instead of collapsed label. Tab labels use plain text (no emojis). Heading hierarchy uses h2 for sections consistently.
- **Preview spinner removed** -- `st.spinner("")` wrapping instant in-memory dataframe eliminated.
- **Imperfection Report conditional** -- only shown when imperfections were actually injected. Collapsed by default.
- **Regenerate spacing** -- added margin before Regenerate button so it doesn't orphan below the divider.

### Changed
- **Emojis removed** from all buttons and tab labels. Icons now use Lucide SVGs via the shared module.
- **Quick Start section removed** from home page -- redundant with sidebar nav and Generate page.
- **Feature cards** on home page use `st.container(border=True)` with centered Lucide icons instead of raw HTML divs.
- **Em-dashes replaced** with double hyphens throughout About page.

## v0.3.1 (2026-06-25)

### Fixed
- **n_rows scoping bug** — CSV/JSON download buttons crashed when user had a cached dataset from a previous session. `n_rows` was defined inside a conditional block but referenced outside. Now stored in `st.session_state`.
- **Permanently skipped MAR detection test** — removed `or True` skip condition. Test now runs and passes.

### Removed
- **`ColumnDef` dataclass** (`generation/models.py`) — defined but never wired into the pipeline. The engine passes raw dicts throughout. Deleted file + 2 orphaned tests.
- **`export_csv` / `export_json`** (`engine.py`) — the Streamlit app handles export directly via `df.to_csv()` / `df.to_json()`. Deleted functions + 1 orphaned test.
- **`KnowledgeGraph.get_columns()`** — only tested, never called by production code. The real query is `get_column_schemas_for_domain()`. Deleted method, rewrote test to use production code path.
- **Dead import** `_generic_schema` in `discovery.py` — `schema_from_kg()` handles fallback internally.

### Changed
- **`_get_config` → `get_config`** — renamed from private to public in `llm/client.py`. Eliminates cross-module private function imports.
- **Redundant no-op removed** — `if isinstance(path, str): path = path` in `_crawl_kaggle`.
- **Duplicate `import tempfile` removed** — from `_crawl_url`.

## v0.3.0 (2026-06-25)

### Added
- **Backend context manager** — `Database.__enter__/__exit__` auto-commits on success, rolls back on exception. Safer transaction boundaries.
- **`ColumnDef` dataclass** — typed DTO for the generation pipeline. `from_dict()` for backward compat with existing dict callers.
- **`KnowledgeGraph.get_column_schemas_for_domain()`** — column query + merging logic moved into the repository. Engine delegates instead of doing raw SQL.
- **Imperfection injection tests** — 2 new tests that actually exercise the injection path (was dead code under test).
- **Domain context on Generate page** — shows domain description and dataset count from SEED_DOMAINS when browsing.
- **76+ tests** (was 70), including Database rollback test, ColumnDef roundtrip tests.
- **Chai4Me support badge** — on main dashboard (not sidebar) + README Support section.

### Fixed
- **Text columns never got nulls** — `inject_nulls()` skipped all non-numeric columns. Now text/object columns accept NaN normally. Booleans and bytes still excluded.
- **numpy datetime64 deprecation** — explicit `np.datetime64(start, "ns")` in datetime generation.

### Changed
- **Progress indicator** — generation now uses `st.status` with phase labels instead of `st.spinner`.
- **Imperfection report** — expandable report with per-column null% and outlier counts after generation.
- **Emoji → Lucide SVGs** — all UI icons replaced with inline Lucide SVGs (MIT-licensed).
- **Tagline sharpened** — "No training. No GPU. No cloud calls." across home page, About page, and README.
- **Copywriting pass** — Quick Start, domain descriptions, and About page copy tightened.

## v0.2.0 (2026-06-24)

### Added
- NL → Schema Discovery pipeline — describe your dataset in plain English
- Gemini support as first-class LLM provider alongside OpenAI
- Schema Knowledge Graph seeded with 21 real datasets across all 10 domains
- Streamlit Cloud deployment support (writable DB path, requirements.txt)
- Full CI pipeline with GitHub Actions

### Changed
- Multi-source crawler now handles Kaggle, HuggingFace, and UCI Archive
- Robust Linux Python compatibility (dict() shim for sqlite3.Row)

## v0.1.0 (2026-06-24)

### Added
- Schema Knowledge Graph with SQLite + FTS5 search (Phase 0, Week 1)
- Multi-source crawler for dataset schema extraction
- Imperfection Fingerprints — null, outlier, skew, and noise profiles for 10 domains (Phase 0, Week 2)
- Core Generator MVP — NumPy/SciPy generation engine with zero training time (Phase 1, Week 3)
- Streamlit UI with domain browser, schema editor, preview, and CSV/JSON export
- Daily cron for knowledge graph enrichment
- Imperfection analyzer and injector pipeline
- AGPL v3 license
- 70+ tests across all modules
- Seed data: Heart Failure, Wine Quality (red + white) datasets
