# Changelog

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
