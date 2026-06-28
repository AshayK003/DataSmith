# Changelog

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
