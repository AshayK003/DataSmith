# Changelog

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
