# ⚒️ DataSmith

> **Realistic synthetic data for dev, testing, and demos.**  
> No training. No GPU. No cloud calls. Pick a domain, edit the schema, generate.

[![CI](https://github.com/AshayK003/DataSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/AshayK003/DataSmith/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-black)](https://docs.astral.sh/uv/)

---

## Why DataSmith?

Most synthetic data tools require ML training (SDV, CTGAN), cloud calls (GPT), or GPU hours. DataSmith takes a different approach: **it generates data from statistical metadata alone**.

A Schema Knowledge Graph stores real dataset schemas (column names, types, distributions) crawled from Kaggle, UCI, and direct URLs. An Imperfection Fingerprint engine captures null patterns, outlier profiles, and noise signatures from real data. The generator composes these into realistic datasets using numpy/scipy — no training, no API calls.

**When to use it:**
- You need test data that *looks real* (not random noise) for QA or staging environments
- You're building a demo and need plausible data across domains
- You want correlated imperfections (nulls, outliers, skewed distributions) in your test data
- You need reproducible datasets (seed-based, deterministic)

**When not to use it:** If you need multi-column correlations or deep statistical fidelity to a specific source dataset — DataSmith generates columns independently. Add SDV ([optional dep](#development)) for correlation support.

---

## Quick Start

```bash
git clone https://github.com/AshayK003/DataSmith.git
cd DataSmith
uv sync
uv run streamlit run app.py
```

Open **http://localhost:8501** → select a domain → edit schema → generate → download.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              Streamlit UI (app.py)            │
│  pages/01_Generate.py → AG Grid → export     │
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
│  KaggleHub │ UCI Archive │ Direct URLs        │
└───────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | What It Does | Key Files |
|-------|-------------|-----------|
| **UI** | Streamlit frontend — domain selection, schema editor (AG Grid), preview, CSV/JSON export | `app.py`, `pages/01_Generate.py`, `pages/02_About.py` |
| **Engine** | Orchestrates generation: schema resolution → data generation → imperfection injection | `generation/engine.py` |
| **Generator** | numpy/scipy-based column generation (numeric, integer, text, boolean, datetime) | `generation/generator.py` |
| **Knowledge Graph** | SQLite-backed schema store with FTS5 search — domains, datasets, column schemas, LLM cache, imperfection profiles | `schema/knowledge_graph.py` |
| **Crawler** | Multi-source schema extraction from Kaggle, UCI Archive, and direct CSV URLs | `schema/crawler.py` |
| **Imperfections** | Statistical analysis + injection of nulls (MCAR/MAR/MNAR), outliers (IQR), noise (rounding), distribution skew | `imperfections/*.py` |
| **LLM** | OpenAI-compatible client with provider abstraction (Groq, OpenRouter, Gemini) — optional NL → schema discovery | `llm/*.py` |

---

## Project Structure

```
📦 datasmith/
├── core/database.py          # SQLite with WAL mode + context manager
├── schema/
│   ├── models.py             # Pydantic models for datasets & columns
│   ├── knowledge_graph.py    # KG CRUD, FTS5 search, domain queries
│   └── crawler.py            # Multi-source schema extraction
├── imperfections/
│   ├── analyzer.py           # Statistical analysis of real data
│   ├── profiles.py           # Domain imperfection fingerprints
│   └── injector.py           # Apply imperfections to clean data
├── generation/
│   ├── generator.py          # numpy/scipy data generation
│   └── engine.py             # Pipeline orchestrator
├── llm/
│   ├── client.py             # OpenAI-compatible API client
│   ├── discovery.py          # NL → schema discovery pipeline
│   └── schemas.py            # Pydantic response models
└── ui/
    ├── components.py         # Shared UI components (header, cards)
    └── icons.py              # Lucide SVG icon library

📦 pages/
├── 01_Generate.py            # Main generation UI
└── 02_About.py               # About / credits page

📦 scripts/
├── crawl_and_analyze.py      # Batch crawl + analyze all domains
├── crawl_schemas.py          # Legacy CLI (crawl only)
└── analyze_domains.py        # Domain fingerprint analysis CLI

📦 tests/                     # 94 tests across 8 files
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `LLM_API_KEY` | No | — | API key for NL → schema discovery |
| `LLM_BASE_URL` | No | Groq endpoint | Base URL for OpenAI-compatible LLM API |
| `LLM_MODEL` | No | Groq's `llama-3.3-70b-versatile` | Model name |
| `LLM_PROVIDER` | No | `groq` | Provider key for routing |

No env vars needed for generation. Only required if using the **Natural Language → Schema** feature (it requires an LLM provider).

Supported providers: `groq`, `openrouter`, `gemini`. All use the same OpenAI-compatible `/chat/completions` format.

---

## Development

### Setup

```bash
uv sync --dev          # Install all deps including dev
uv run pytest          # Run all tests
```

### Code Flow (Adding a New Domain)

1. **Add seed data** in `schema/crawler.py` → `SEED_DATASETS` dict with domain name and dataset URLs
2. **Run crawler** `uv run python scripts/crawl_and_analyze.py` to populate the KG
3. **Run analyzer** `uv run python scripts/analyze_domains.py --domain <name>` to extract imperfection fingerprints
4. **Test generation** — the new domain appears in the Streamlit UI dropdown automatically

### Code Flow (Adding a New Column Type)

1. Add a sampler function in `generation/generator.py` (e.g., `_sample_mytype`)
2. Add the type→sampler mapping in `generate_column()`'s dispatch logic
3. Add the type option in `pages/01_Generate.py`'s AG Grid type dropdown
4. Add tests in `tests/test_generation.py`

### Linting

```bash
uv run flake8 datasmith/ tests/ --max-line-length=100
```

---

## Testing

```bash
uv run pytest                    # All tests
uv run pytest -v                 # Verbose output
uv run pytest tests/test_generation.py --tb=long  # Detailed traceback
uv run pytest -k "edge"          # Only edge-case tests
uv run pytest --cov=datasmith    # Coverage report
```

**94 tests** across 8 files. Test structure mirrors source structure:

| Test File | What It Covers | Tests |
|-----------|---------------|-------|
| `test_generation.py` | Core generator, pipeline, schema resolution | 24 |
| `test_generation_edge.py` | Edge cases: empty schema, degenerate distributions, reversed ranges | 4 |
| `test_imperfections.py` | Analyzer + injector: nulls, outliers, noise, skew, profiles | 30 |
| `test_injector_edge.py` | Injector edge cases: missing columns, self-referencing correlations, all-NaN | 6 |
| `test_knowledge_graph.py` | KG CRUD: domains, datasets, columns, cache, profiles, migration | 16 |
| `test_llm.py` | LLM discovery: caching, parsing, response → schema mapping | 10 |
| `test_llm_client.py` | LLM client: config, error handling, timeouts | 4 |

### Writing Tests

- Use `np.random.default_rng(seed)` for reproducible random output
- Test with edge values: `0`, `NaN`, `None`, empty strings, reversed date ranges
- Follow existing patterns — pytest classes with descriptive test method names

---

## Deployment

### Streamlit Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo
3. Set **Python version** to `3.11` in Advanced Settings
4. Set **Command**: `streamlit run app.py`
5. (Optional) Set secrets via Streamlit Cloud dashboard if using LLM discovery

The app writes the SQLite database to Streamlit Cloud's ephemeral storage. Data persists within a session. For persistent storage, set `DATASMITH_DB_PATH` to an external path.

### Render (Free Tier)

```bash
# Build command
uv sync --dev

# Start command
uv run streamlit run app.py --server.port $PORT --server.headless true
```

---

## Common Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `streamlit-aggrid` import error | Missing dependency | `uv sync` (reinstalls all deps) |
| LLM discovery returns nothing | Missing or invalid `LLM_API_KEY` | Set API key in environment, verify provider endpoint |
| No domains in dropdown | KG not seeded | `uv run python scripts/crawl_and_analyze.py` |
| "Generation failed" error | Invalid schema params (min > max, negative std) | Check column bounds in the schema editor |
| AG Grid shows no data after edit | Streamlit re-run reset state | Grid auto-persists in session_state — should be stable |
| Crawler fails on all URLs | Network issue or stale dataset URLs | Check `SEED_DATASETS` in `crawler.py`, replace dead URLs |
| Tests fail with sqlite3 error | Test isolation issue | Check for concurrent test DB writes — each test should use a temp DB |

---

## Security Notes

- **No authentication** — this is a local/single-user tool. Do not expose to the open internet without auth middleware.
- **LLM keys** — API keys live in environment variables, never in code or session state
- **CSV output** — sanitized against formula injection (cells starting with `=`, `+`, `-`, `@` are prefixed with `'`)
- **AG Grid** — JavaScript execution disabled (`allow_unsafe_jscode=False`)
- **Input validation** — column names are sanitized (alphanumeric + spaces, capped at 128 chars). LLM input is stripped of control characters and isolated in XML tags.

---

## Contributing

### Principles

- **Default to minimal deps.** If stdlib can do it in ≤50 lines, don't add a package.
- **Default to numpy.** scipy is fine where needed. No PyTorch in core (optional `sdv` dep is the exception).
- **Test with edge values**, not just standard cases. A NaN in a probability mask can be a silent bug.
- **One-file changes** are preferred over multi-file refactors for bug fixes.
- **Every fix gets a test.** If it can break, prove it can't break again.

### PR Checklist

- [ ] `uv run pytest` passes
- [ ] `uv run flake8 datasmith/ tests/ --max-line-length=100` passes
- [ ] New functionality has tests covering: normal path, edge values, error state
- [ ] No new dependencies without discussion
- [ ] README unchanged unless behavior changed

### Code Review Focus

Ordered by importance: correctness → edge cases → test coverage → readability → performance → style.

---

## License

**AGPL v3** — free to use, share, and modify. Cannot be used in closed-source commercial products. See [LICENSE](LICENSE).

---

## Support

If DataSmith saves you time or helps your project:

<a href="https://chai4.me/ashaykushwaha003" target="_blank" title="Support ashaykushwaha003 on Chai4Me" style="display:inline-flex;flex-direction:column;align-items:center;justify-content:center;background:#ffffff;padding:8px 32px;border-radius:16px;text-decoration:none;border:1px solid #e5e7eb;box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);transition:transform 0.2s;"><img src="https://chai4.me/icons/wordmark.png" alt="Chai4Me" style="height:32px;object-fit:contain;"/></a>
