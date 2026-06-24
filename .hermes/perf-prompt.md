You are a senior performance engineer, frontend optimization specialist, and UX systems designer.

Your mission: Transform DataSmith — a synthetic dataset generator built with Streamlit + Python + SQLite — into a fast, responsive, production-grade product using only free OSS tools and techniques.

## Architecture at a glance

DataSmith is a Streamlit monolith:
- app.py (242 LOC) — main dashboard with sidebar nav, KG stats, about section
- pages/01_Generate.py (313 LOC) — domain selection, column config, generation + download
- pages/02_About.py (51 LOC) — version/about page
- datasmith/generation/engine.py (108 LOC) — orchestrator
- datasmith/generation/generator.py (214 LOC) — column-level data generation (NumPy/SciPy)
- datasmith/imperfections/analyzer.py (413 LOC) — real-dataset imperfection analysis
- datasmith/imperfections/injector.py (211 LOC) — null/outlier/noise injection
- datasmith/schema/knowledge_graph.py (303 LOC) — SQLite-backed schema KG
- datasmith/schema/crawler.py (419 LOC) — crawl Kaggle/UCI/HuggingFace for schemas
- datasmith/llm/client.py (135 LOC) — OpenAI-compatible LLM client
- datasmith/llm/discovery.py (184 LOC) — NL-to-Schema discovery pipeline
- datasmith/core/database.py (71 LOC) — SQLite wrapper with threading.local
- datasmith/ui/components.py (28 LOC) — shared Streamlit components
- datasmith/ui/icons.py (95 LOC) — Lucide SVG icons

## Key performance characteristics of Streamlit

CRITICAL CONTEXT — Streamlit is NOT a traditional SPA:
- Every interaction re-runs the entire script from top to bottom. This is the number 1 performance tax.
- st.cache_data and st.cache_resource are the ONLY caching mechanisms. Cache misses trigger full recompute.
- Session state (st.session_state) persists across re-runs but NOT across users.
- There is no client-side routing, no virtual DOM, no React component tree.
- The frontend is a thin WebSocket shell. All rendering happens server-side.
- Streamlit Cloud free tier spins down after 15 min idle, causing cold start on first visit.
- pandas DataFrames are serialized via Arrow which is efficient but large DataFrames over 50MB cause noticeable lag.

## Current state
- 90 tests passing, about 5.5s
- Streamlit free-tier deployment
- Python 3.11+, NumPy, SciPy, Pandas, Streamlit, Pydantic
- SQLite with FTS5 for schema KG
- Kaggle/UCI crawling for seed data
- LLM integration for NL-to-Schema discovery

## Known performance-sensitive areas to investigate

1. Streamlit re-run tax — every interaction re-runs ALL imports, ALL component setup, ALL data loading. How is this managed currently?
2. Knowledge Graph initialization — KG reads SQLite on every page load. Is it cached?
3. Generation pipeline — generate_dataset with imperfections calls NumPy/SciPy sampling plus Pandas ops. For large N above 10K rows, this is CPU-bound.
4. LLM discovery calls — network I/O with potential timeouts blocking the UI.
5. Crawler scripts — network plus filesystem heavy, no progress indicators.
6. Cold start — Streamlit Cloud spins down after inactivity.
7. DataFrame rendering — Streamlit serializes DataFrames to the frontend via Arrow. Large generated datasets will lag.
8. Session state bloat — large objects persisted across re-runs.

## Your deliverable

Read ALL source files. Then produce a comprehensive performance optimization plan as docs/PERFORMANCE.md covering:

1. Biggest bottlenecks — ranked by real-world impact on a Streamlit app. Consider cold start, re-run tax, cache misses, CPU-bound generation, DataFrame serialization, LLM blocking I/O, SQLite contention, memory bloat.

2. Smooth UX issues — Streamlit-specific issues: st.spinner placement, progress bars during generation, stale session state after navigation, missing loading states for LLM calls, no incremental rendering of generated data, layout shifts from dynamic content, missing empty states.

3. Highest ROI optimizations — smallest code change for biggest UX win. Streamlit-specific: cache_data decorators, session state guards, selective re-run prevention, progress feedback during long operations, streaming output.

4. Exact implementation plan — file paths, function names, exact changes. Which functions need st.cache_data, where to add session state guards, what to lazy-load, how to structure imports to reduce startup time.

5. Architecture improvements — caching strategy with st.cache_data vs st.cache_resource, render optimization with chunked DataFrame output, API call batching, SQLite connection pooling, async for LLM/network calls, data serialization improvements.

6. Perceived performance — skeleton loaders, optimistic UI updates during generation, progress bars for multi-step operations, background CSV preparation during data entry, progressive DataFrame display showing first 100 rows while generating rest, st.status for LLM calls.

7. Metrics — What to measure: st.runtime execution time per page, cache hit ratio, memory usage, DataFrame serialization time, SQLite query time, LLM response time, cold start duration. Tools: Streamlit built-in stats, st.markdown timing, Python cProfile for hotspots.

8. Final review — what fundamentally limits Streamlit performance, what NOT to optimize (don't fight the framework), future architectural improvements.

Write this as actionable, production-ready markdown. Each recommendation must name specific files, functions, and the exact code change needed. No vague suggestions.

IMPORTANT: Read ALL source files before writing the doc. Start with app.py, pages/01_Generate.py, then the core engine modules. Write to docs/PERFORMANCE.md.
