# ⚒️ DataSmith

> **Generate realistic synthetic datasets for development, testing, and demos.**
> Powered by a Schema Knowledge Graph (real dataset schemas) + Domain Imperfection Fingerprints (statistical patterns from real data).

[![CI](https://github.com/AshayK003/DataSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/AshayK003/DataSmith/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-black)](https://docs.astral.sh/uv/)

## Quick Start

```bash
# Install
git clone https://github.com/AshayK003/DataSmith.git
cd DataSmith
uv sync
uv run streamlit run app.py

# Or seed the knowledge graph first
uv run python scripts/crawl_and_analyze.py
```

Open http://localhost:8501 → select a domain → edit schema → generate → download.

## Features

- **10 domain profiles** — e-commerce, healthcare, finance, education, social-media, IoT, real-estate, transportation, energy, manufacturing
- **Realistic imperfections** — nulls (MCAR/MAR/MNAR), outliers (IQR-based), noise (rounding), distribution-skew patterns — all extracted from real datasets
- **Schema Knowledge Graph** — real dataset schemas crawled from Kaggle, HuggingFace, and UCI Archive, stored in SQLite with FTS5 search
- **No training needed** — data generated from statistical metadata using numpy/scipy. Instant generation, zero GPU, zero API calls
- **Streamlit UI** — domain selector, editable schema table, preview, CSV/JSON export
- **Lightweight** — pure Python, 8 core deps, no PyTorch/SDV required

## Architecture

```
┌──────────────────────────────────────────────┐
│              Streamlit UI (app.py)            │
│  pages/01_Generate.py → preview → export     │
└────────────┬─────────────────────────────────┘
             │ calls
┌────────────▼─────────────────────────────────┐
│          Generation Engine (engine.py)        │
│  schema_from_kg() → generate_dataset()       │
│           │                │                  │
│      ┌────▼────┐     ┌────▼──────┐           │
│      │ Schema  │     │ Generator │           │
│      │   KG    │     │ (numpy/   │           │
│      │(SQLite) │     │  scipy)   │           │
│      └─────────┘     └────┬──────┘           │
│                           │                   │
│                     ┌────▼──────┐            │
│                     │ Injector  │            │
│                     │(nulls,    │            │
│                     │ outliers, │            │
│                     │ noise)    │            │
│                     └───────────┘            │
└──────────────────────────────────────────────┘
         ▲
         │ crawls
┌────────┴──────────────────────────────────────┐
│           Schema Crawler (crawler.py)          │
│  KaggleHub │ HuggingFace │ UCI Archive        │
└───────────────────────────────────────────────┘
```

## Project Structure

```
📦 datasmith/
├── core/database.py        # SQLite with WAL mode
├── schema/
│   ├── models.py           # Pydantic models
│   ├── knowledge_graph.py  # KG CRUD + FTS5 search
│   └── crawler.py          # Multi-source schema extraction
├── imperfections/
│   ├── analyzer.py          # Statistical analysis of real data
│   ├── profiles.py          # Domain imperfection profiles
│   └── injector.py          # Apply imperfections to clean data
└── generation/
    ├── generator.py         # numpy/scipy data generation
    └── engine.py            # Pipeline orchestrator

📦 pages/
├── 01_Generate.py           # Main generation UI
└── 02_About.py              # About page

📜 scripts/
├── crawl_schemas.py         # Legacy schema crawler CLI
├── crawl_and_analyze.py     # Batch crawl + analyze all domains
└── analyze_domains.py       # Domain fingerprint analysis

📜 tests/                    # 60+ tests, run via: uv run pytest
📜 app.py                    # Streamlit entry point
```

## Development

```bash
# Setup
uv sync --dev
uv run pytest tests/ -v

# Run analysis on a specific domain
uv run python scripts/analyze_domains.py --domain healthcare

# Full crawl + analysis (URL sources only)
uv run python scripts/crawl_and_analyze.py --source url

# Lint
uv run flake8 datasmith/ tests/ --max-line-length=100
```

## License

**AGPL v3** — free to use, share, and modify. Cannot be used in closed-source commercial products. See [LICENSE](LICENSE).

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 0 | Schema Knowledge Graph + Imperfection Fingerprints | ✅ Complete |
| Phase 1 | Core Generator MVP (Streamlit app + generation) | ✅ Complete |
| Phase 2 | Community Schema Library + SDV integration | 🔜 Planned |
