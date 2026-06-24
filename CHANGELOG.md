# Changelog

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
