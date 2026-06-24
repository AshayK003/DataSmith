# DataSmith

> Free open-source synthetic dataset generator with a real moat.

Describe any dataset in plain English → auto-discovers real schemas from a knowledge graph of thousands of real datasets → generates realistic synthetic data with domain-specific imperfections.

**Status:** Phase 0 — Moat Engine (Weeks 1–4). Schema crawler + imperfection fingerprinting in progress.

## Architecture

```
Week 1-4: Schema Knowledge Graph + Imperfection Fingerprints (data moat)
Week 5-8: Streamlit generation UI (thin layer over moat)
Week 9-12: Community schema library (network moat)
```

**Stack:** Python, Streamlit, SQLite (FTS5), SDV, Frictionless, KaggleHub, NumPy/SciPy, Pydantic

## License

AGPL v3 — free to use, share, and modify. Cannot be used in closed-source commercial products.
